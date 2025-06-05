import torch
import pickle
import pytorch_lightning as pl
from typing import Any, Dict
from yacs.config import CfgNode
from ..utils.geometry import perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_smal_head
from .heads.classifier_head import ClassTokenHead
from amr.models.smal_warapper import SMAL
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, ShapePriorLoss, PosePriorLoss, SupConLoss
log = get_pylogger(__name__)


class AMR(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup AMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)

        # Create SMAL head
        self.smal_head = build_smal_head(cfg)

        # Instantiate SMAL model
        smal_model_path = cfg.SMAL.MODEL_PATH
        with open(smal_model_path, 'rb') as f:
            smal_cfg = pickle.load(f, encoding="latin1")
        self.smal = SMAL(**smal_cfg)

        self.class_token_head = ClassTokenHead(**cfg.MODEL.get("CLASS_TOKEN_HEAD", dict()))

        # Create discriminator
        self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smal_parameter_loss = ParameterLoss()
        self.shape_prior_loss = ShapePriorLoss(path_prior=cfg.SMAL.SHAPE_PRIOR_PATH)
        self.pose_prior_loss = PosePriorLoss(path_prior=cfg.SMAL.POSE_PRIOR_PATH)
        self.supcon_loss = SupConLoss()

        self.register_buffer('initialized', torch.tensor(False))

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        conditioning_feats, cls = self.backbone(x[:, :, :, 32:-32])  # [256, 192]
        pred_smal_params, pred_cam, _ = self.smal_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output = {}
        output['cls_token'] = cls
        output['cls_feats'] = self.class_token_head(cls) if self.cfg.MODEL.BACKBONE.get("USE_CLS", False) else None

        output['pred_cam'] = pred_cam
        output['pred_smal_params'] = {k: v.clone() for k, v in pred_smal_params.items()}

        # Compute camera translation
        focal_length = batch['focal_length']
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)], dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_smal_params['global_orient'] = pred_smal_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smal_params['pose'] = pred_smal_params['pose'].reshape(batch_size, -1, 3, 3)
        pred_smal_params['betas'] = pred_smal_params['betas'].reshape(batch_size, -1)
        smal_output = self.smal(**pred_smal_params, pose2rot=False)

        pred_keypoints_3d = smal_output.joints
        pred_vertices = smal_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
                
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

