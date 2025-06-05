import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import pickle as pkl
from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder


def build_smal_head(cfg):
    smal_head_type = cfg.MODEL.SMAL_HEAD.get('TYPE', 'amr')
    if smal_head_type == 'transformer_decoder':
        return SMALTransformerDecoderHead(cfg)
    else:
        raise ValueError('Unknown SMAL head type: {}'.format(smal_head_type))


class SMALTransformerDecoderHead(nn.Module):
    """ Cross-attention based SMAL Transformer decoder
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.SMAL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMAL.NUM_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.SMAL_HEAD.get('TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = {**transformer_args, **dict(cfg.MODEL.SMAL_HEAD.TRANSFORMER_DECODER)}
        
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim = transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 41)
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.SMAL_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        init_pose = torch.zeros(size=(1, npose), dtype=torch.float32)
        init_betas = torch.zeros(size=(1, 41), dtype=torch.float32)
        init_cam = torch.tensor([[0.9, 0, 0]], dtype=torch.float32)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_pose = self.init_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.cfg.MODEL.SMAL_HEAD.get('IEF_ITERS', 3)):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_pose, pred_betas, pred_cam], dim=1)[:, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)  # (B, C)

            # Readout from token_out
            pred_pose = self.decpose(token_out) + pred_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam
            pred_pose_list.append(pred_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_smal_params_list = {}
        pred_smal_params_list['pose'] = torch.cat(
            [joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_pose_list], dim=0)
        pred_smal_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_smal_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_pose = joint_conversion_fn(pred_pose).view(batch_size, self.cfg.SMAL.NUM_JOINTS + 1, 3, 3)

        pred_smal_params = {'global_orient': pred_pose[:, [0]],
                            'pose': pred_pose[:, 1:],
                            'betas': pred_betas,
                            }
        return pred_smal_params, pred_cam, pred_smal_params_list



