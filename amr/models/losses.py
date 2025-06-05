import torch
import torch.nn as nn
import numpy as np
import pickle
from pytorch3d.transforms import matrix_to_axis_angle
import torch.nn.functional as F


class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1, 2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 0):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1, 2))
        return loss.sum()


class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMAL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute SMAL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth MANO parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        mask = torch.ones_like(pred_param, device=pred_param.device, dtype=pred_param.dtype)
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims - 1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param*mask, gt_param*mask))
        return loss_param.sum()


class PosePriorLoss(nn.Module):
    def __init__(self, path_prior):
        super(PosePriorLoss, self).__init__()
        with open(path_prior, "rb") as f:
            data_prior = pickle.load(f, encoding="latin1")

        self.register_buffer("mean_pose", torch.from_numpy(data_prior["mean_pose"]).float())
        self.register_buffer("precs", torch.from_numpy(np.array(data_prior["pic"])).float())

        use_index = np.ones(105, dtype=bool)
        use_index[:3] = False  # global rotation set False
        self.register_buffer("use_index", torch.from_numpy(use_index).float())

    def forward(self, x, has_gt):
        """
        Args:
            x: (batch_size, 35, 3, 3)
            has_gt: has pose?
        Returns:
            pose prior loss
        """
        if has_gt.sum() == len(has_gt):
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        has_gt = has_gt.type(torch.bool)
        x = x[~has_gt]
        x = matrix_to_axis_angle(x.reshape(-1, 3, 3))
        delta = x.reshape(-1, 35*3) - self.mean_pose
        loss = torch.tensordot(delta, self.precs, dims=([1], [0])) * self.use_index
        return (loss ** 2).mean()


class ShapePriorLoss(nn.Module):
    def __init__(self, path_prior):
        super(ShapePriorLoss, self).__init__()
        with open(path_prior, "rb") as f:
            data_prior = pickle.load(f, encoding="latin1")

        model_covs = np.array(data_prior["cluster_cov"])  # shape: (5, 41, 41)
        inverse_covs = np.stack(
            [np.linalg.inv(model_cov + 1e-5 * np.eye(model_cov.shape[0])) for model_cov in model_covs],
            axis=0)
        prec = np.stack([np.linalg.cholesky(inverse_cov) for inverse_cov in inverse_covs], axis=0)

        self.register_buffer("betas_prec", torch.FloatTensor(prec))
        self.register_buffer("mean_betas", torch.FloatTensor(data_prior["cluster_means"]))

    def forward(self, x, category, has_gt):
        """
        Args:
            x: predicted betas (batch_size, 41)
            category: animal category (batch_size,)
            has_gt: has shape?
        Returns:
            shape prior loss
        """
        if has_gt.sum() == len(has_gt):
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        has_gt = has_gt.type(torch.bool)
        x, category = x[~has_gt], category[~has_gt]
        delta = (x - self.mean_betas[category.long()])  # [batch_size, 41]
        loss = []
        for x0, c0 in zip(delta, category):
            loss.append(torch.tensordot(x0, self.betas_prec[c0], dims=([0], [0])))
        loss = torch.stack(loss, dim=0)
        return (loss ** 2).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = torch.stack((features, features), dim=1)
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
