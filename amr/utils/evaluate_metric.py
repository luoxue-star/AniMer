import torch
import numpy as np
import open3d as o3d
from typing import Dict, List, Union
from pytorch3d.transforms import axis_angle_to_matrix


def compute_scale_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a scale transform (s) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3).
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the scale transformation.
    """

    # 1. Remove mean.
    mu1 = S1.mean(dim=1, keepdim=True)
    mu2 = S2.mean(dim=1, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1 ** 2).sum(dim=(1, 2), keepdim=True)

    # 3. Compute scale.
    scale = (X2 * X1).sum(dim=(1, 2), keepdim=True) / var1

    # 4. Apply scale transform.
    S1_hat = scale * X1 + mu2

    return S1_hat


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1 ** 2).sum(dim=(1, 2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1.float(), X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K.float())
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1).float()
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U.float(), Vh.float()).float()))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale * torch.matmul(R.float(), mu1.float())

    # 7. Error:
    S1_hat = scale * torch.matmul(R.float(), S1.float()).float() + t

    return S1_hat.permute(0, 2, 1)


def pointcloud(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points)
    pcd.points = points
    return pcd


class Evaluator:
    def __init__(self, smal_model, image_size: int=256, pelvis_ind: int = 7):
        self.pelvis_ind = pelvis_ind
        self.smal_model = smal_model
        self.image_size = image_size
    
    def compute_pck(self, output: Dict, batch: Dict, pck_threshold: Union[List, None]):
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().cpu()
        gt_keypoints_2d = batch['keypoints_2d'].detach().cpu()
        self.pck_threshold_list = []
        
        pred_keypoints_2d = (pred_keypoints_2d + 0.5) * self.image_size  # * batch['bbox_expand_factor'].detach().cpu().numpy().reshape(-1, 1, 1)
        conf = gt_keypoints_2d[:, :, -1]
        gt_keypoints_2d = (gt_keypoints_2d[:, :, :-1] + 0.5) * self.image_size  # * batch['bbox_expand_factor'].detach().cpu().numpy().reshape(-1, 1, 1)
        if pck_threshold is not None:
            for i in range(len(pck_threshold)):
                self.pck_threshold_list.append(torch.tensor([pck_threshold[i]] * len(pred_keypoints_2d), dtype=torch.float32))

        pcks = []
        seg_area = torch.sum(batch['mask'].detach().cpu().reshape(batch['mask'].shape[0], -1), dim=-1).unsqueeze(-1)
        total_visible = torch.sum(conf, dim=-1)
        for th in self.pck_threshold_list:
            dist = torch.norm(pred_keypoints_2d - gt_keypoints_2d, dim=-1)

            hits = (dist / torch.sqrt(seg_area)) < th.unsqueeze(1)
            pck = torch.sum(hits.float() * conf, dim=-1) / total_visible
            pcks.append(pck.numpy().tolist())
        return torch.mean(torch.tensor(pcks), dim=1)

    def compute_pa_mpjpe(self, pred_joints, gt_joints):
        S1_hat = compute_similarity_transform(pred_joints, gt_joints)
        pa_mpjpe = torch.sqrt(((S1_hat - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * 1000
        return pa_mpjpe.mean()

    def compute_pa_mpvpe(self, gt_vertices: torch.Tensor, pred_vertices: torch.Tensor):
        batch_size = pred_vertices.shape[0]
        S1_hat = compute_similarity_transform(pred_vertices, gt_vertices)
        pa_mpvpe = torch.sqrt(((S1_hat - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * 1000
        return pa_mpvpe.mean()

    def eval_3d(self, output: Dict, batch: Dict):
        """
        Evaluate current batch
        Args:
            output: model output
            batch: model input
        Returns: evaluate metric
        """
        if batch['has_smal_params']["betas"].sum() == 0:
            return 0., 0., 0., [0., 0.], 0.

        pred_keypoints_3d = output["pred_keypoints_3d"].detach()
        pred_keypoints_3d = pred_keypoints_3d[:, None, :, :]
        batch_size = pred_keypoints_3d.shape[0]
        num_samples = pred_keypoints_3d.shape[1]
        gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)
        gt_vertices = self.smal_forward(batch)
        
        # Align predictions and ground truth such that the pelvis location is at the origin
        pred_keypoints_3d -= pred_keypoints_3d[:, :, [self.pelvis_ind]]
        gt_keypoints_3d -= gt_keypoints_3d[:, :, [self.pelvis_ind]]

        pa_mpjpe = self.compute_pa_mpjpe(pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3),
                                         gt_keypoints_3d.reshape(batch_size * num_samples, -1, 3))
        pa_mpvpe = self.compute_pa_mpvpe(gt_vertices, output['pred_vertices'])
        return pa_mpjpe, pa_mpvpe
    
    def eval_2d(self, output: Dict, batch: Dict, pck_threshold: List[float]=[0.10, 0.15]):
        pck = self.compute_pck(output, batch, pck_threshold=pck_threshold)
        auc = self.compute_auc(batch, output)
        return pck.tolist(), auc
    
    def compute_auc(self, batch: Dict, output: Dict, threshold_min: int=0.0, threshold_max: int=1.0, steps: int=100):
        thresholds = np.linspace(threshold_min, threshold_max, steps)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)
        pck_curve = []
        for th in thresholds:
            pck_curve.append(self.compute_pck(output, batch, [th]))
        pck_curve = torch.tensor(pck_curve).tolist()
        auc = np.trapz(pck_curve, thresholds)
        auc /= norm_factor
        return auc

    def smal_forward(self, batch: Dict):
        batch_size = batch['img'].shape[0]
        smal_params = batch['smal_params']
        smal_params['global_orient'] = axis_angle_to_matrix(smal_params['global_orient'].reshape(batch_size, -1)).unsqueeze(1)
        smal_params['pose'] = axis_angle_to_matrix(smal_params['pose'].reshape(batch_size, -1, 3))
        smal_params = {k: v.cuda() for k, v in smal_params.items()}
        with torch.no_grad():
            smal_output = self.smal_model(**smal_params)
        vertices = smal_output.vertices
        return vertices
