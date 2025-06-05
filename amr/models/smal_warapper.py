import json
from torch import nn
import torch
import numpy as np
import pickle
import cv2
from typing import Optional, Tuple, NewType
from dataclasses import dataclass
import smplx
from smplx.lbs import vertices2joints, lbs
from smplx.utils import MANOOutput, to_tensor, ModelOutput
from smplx.vertex_ids import vertex_ids

Tensor = NewType('Tensor', torch.Tensor)
keypoint_vertices_idx = [[1068, 1080, 1029, 1226], [2660, 3030, 2675, 3038], [910], [360, 1203, 1235, 1230],
                         [3188, 3156, 2327, 3183], [1976, 1974, 1980, 856], [3854, 2820, 3852, 3858], [452, 1811],
                         [416, 235, 182], [2156, 2382, 2203], [829], [2793], [60, 114, 186, 59],
                         [2091, 2037, 2036, 2160], [384, 799, 1169, 431], [2351, 2763, 2397, 3127],
                         [221, 104], [2754, 2192], [191, 1158, 3116, 2165],
                         [28, 1109, 1110, 1111, 1835, 1836, 3067, 3068, 3069],
                         [498, 499, 500, 501, 502, 503], [2463, 2464, 2465, 2466, 2467, 2468],
                         [764, 915, 916, 917, 934, 935, 956], [2878, 2879, 2880, 2897, 2898, 2919, 3751],
                         [1039, 1845, 1846, 1870, 1879, 1919, 2997, 3761, 3762],
                         [0, 464, 465, 726, 1824, 2429, 2430, 2690]]

name2id35 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1,
             'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15,
             'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11,
             'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27,
             'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0, 'LEar': 33, 'REar': 34, 'EndNose': 35, 'Chin': 36,
             'RightEarTip': 37, 'LeftEarTip': 38, 'LeftEye': 39, 'RightEye': 40}

@dataclass
class SMALOutput(ModelOutput):
    betas: Optional[Tensor] = None
    pose: Optional[Tensor] = None


class SMALLayer(nn.Module):
    def __init__(self, num_betas=41, **kwargs):
        super().__init__()
        self.num_betas = num_betas
        self.register_buffer("shapedirs", torch.from_numpy(np.array(kwargs['shapedirs'], dtype=np.float32))[:, :, :num_betas]) # [3889, 3, 41]
        self.register_buffer("v_template", torch.from_numpy(np.array(kwargs['v_template']).astype(np.float32)))  # [3889, 3]
        self.register_buffer("posedirs", torch.from_numpy(np.array(kwargs['posedirs'], dtype=np.float32)).reshape(-1,
                                                                                                 34*9).T)  # [34*9, 11667]
        self.register_buffer("J_regressor", torch.from_numpy(kwargs['J_regressor'].toarray().astype(np.float32)))  # [33, 3389]
        self.register_buffer("lbs_weights", torch.from_numpy(np.array(kwargs['weights'], dtype=np.float32)))  # [3889, 33]
        self.register_buffer("faces", torch.from_numpy(np.array(kwargs['f'], dtype=np.int32)))  # [7774, 3]

        kintree_table = kwargs['kintree_table']
        self.register_buffer("parents", torch.from_numpy(kintree_table[0].astype(np.int32)))

    def forward(
            self,
            betas: Optional[Tensor] = None,
            global_orient: Optional[Tensor] = None,
            pose: Optional[Tensor] = None,
            transl: Optional[Tensor] = None,
            return_verts: bool = True,
            return_full_pose: bool = False,
            **kwargs):
        """
        Args:
            betas: [batch_size, 10]
            global_orient: [batch_size, 1, 3, 3]
            pose: [batch_size, num_joints, 3, 3]
            transl: [batch_size, num_joints, 3]
            return_verts:
            return_full_pose:
            **kwargs:
        Returns:
        """
        device, dtype = betas.device, betas.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        else:
            batch_size = global_orient.shape[0]
        if pose is None:
            pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 34, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat([global_orient, pose], dim=1)
        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False)

        if transl is not None:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = SMALOutput(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            pose=pose,
            transl=transl,
            full_pose=full_pose if return_full_pose else None,
        )
        return output


class SMAL(SMALLayer):
    def __init__(self, **kwargs):
        super(SMAL, self).__init__(**kwargs)

    def forward(self, *args, **kwargs):
        smal_output = super(SMAL, self).forward(**kwargs)

        keypoint = []
        for kp_v in keypoint_vertices_idx:
            keypoint.append(smal_output.vertices[:, kp_v, :].mean(dim=1))
        smal_output.joints = torch.stack(keypoint, dim=1)
        return smal_output


