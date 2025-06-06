import torch
import pickle
import pytorch_lightning as pl
from typing import Any, Dict
from yacs.config import CfgNode
from torchvision.utils import make_grid
from ..utils.geometry import perspective_projection, aa_to_rotmat
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_smal_head
from .heads.classifier_head import ClassTokenHead
from ..utils import MeshRenderer
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
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            state_dict = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu', weights_only=True)['state_dict']
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)

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

        # Setup renderer for visualization
        if init_renderer:
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smal.faces.numpy())
        else:
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def get_parameters(self):
        all_params = list(self.smal_head.parameters())
        all_params += list(self.backbone.parameters())
        all_params += list(self.class_token_head.parameters())
        return all_params

    def configure_optimizers(self):
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]
        
        optimizer = torch.optim.AdamW(params=param_groups,
                                      # lr=self.cfg.TRAIN.LR,
                                      weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        if self.cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0) > 0:
            optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                               lr=self.cfg.TRAIN.LR,
                                               weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        else:
            return optimizer,

        return optimizer, optimizer_disc

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
        output['cls_feats'] = self.class_token_head(cls)

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

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        
        pred_smal_params = output['pred_smal_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_smal_params['pose'].shape[0]
        device = pred_smal_params['pose'].device
        dtype = pred_smal_params['pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smal_params = batch['smal_params']
        gt_mask = batch['mask']
        has_smal_params = batch['has_smal_params']
        is_axis_angle = batch['smal_params_is_axis_angle']
        has_mask = batch['has_mask']

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # Compute loss on SMAL parameters
        loss_smal_params = {}
        for k, pred in pred_smal_params.items():
            gt = gt_smal_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_smal_params[k]
            if k == "betas":
                loss_smal_params[k] = self.smal_parameter_loss(pred.reshape(batch_size, -1),
                                                               gt.reshape(batch_size, -1),
                                                               has_gt) + \
                                      self.shape_prior_loss(pred, batch["category"], has_gt)
            else:
                loss_smal_params[k] = self.smal_parameter_loss(pred.reshape(batch_size, -1),
                                                               gt.reshape(batch_size, -1),
                                                               has_gt) + \
                                      self.pose_prior_loss(torch.cat((pred_smal_params["global_orient"],
                                                                      pred_smal_params["pose"]),
                                                                      dim=1), has_gt) / 2.
        loss_supcon = self.supcon_loss(output['cls_feats'], labels=batch['category'])
        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d + \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d + \
               sum([loss_smal_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smal_params]) + \
               self.cfg.LOSS_WEIGHTS['SUPCON'] * loss_supcon

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach(),
                      loss_supcon=loss_supcon.detach(),
                      )

        for k, v in loss_smal_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        return loss
    
    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            pose (torch.Tensor): Regressed pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = pose.shape[0]
        gt_pose = batch['pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat.detach(), gt_betas.detach())
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()    

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True,
                            write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        # mul std then add mean
        images = (images) * (torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1))
        images = (images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1))

        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode + '/' + loss_name, val.detach().item(), step_count)
            if train is False:
                for metric_name, val in output['metric'].items():
                    summary_writer.add_scalar(mode + '/' + metric_name, val, step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               self.cfg.SMAL.get("FOCAL_LENGTH", 1000),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               )
        predictions = make_grid(predictions, nrow=5, padding=2)
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def training_step(self, batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            batch (Dict): Dictionary containing {'img', 'mask', 'keypoints_2d', 'keypoints_3d', 'orig_keypoints_2d',
                                                'box_center', 'box_size', 'img_size', 'smal_params',
                                                'smal_params_is_axis_angle', '_trans', 'imgname', 'focal_length'}
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = batch['img']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0) > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smal_params = output['pred_smal_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0) > 0:
            disc_out = self.discriminator(pred_smal_params['pose'].reshape(batch_size, -1),
                                          pred_smal_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL,
                                                error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # For compatibility
        # if self.cfg.LOSS_WEIGHTS.ADVERSARIAL == 0:
        #     optimizer.param_groups[0]['capturable'] = True
        
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0) > 0:
            loss_disc = self.training_step_discriminator(batch['smal_params'],
                                                         pred_smal_params['pose'].reshape(batch_size, -1),
                                                         pred_smal_params['betas'].reshape(batch_size, -1),
                                                         optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
                 batch_size=batch_size, sync_dist=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        pass
