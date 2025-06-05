from .amr import AMR


def load_amr(checkpoint_path):
    from pathlib import Path
    from ..configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / '.hydra/config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if (model_cfg.MODEL.BACKBONE.TYPE == 'dinov2') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model.cfg.MODEL.IMAGE_SIZE == 252, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL_IMAGE_SIZE}) should be 252 for dino backbone"
        model_cfg.MODEL_BBOX_SHAPE = [252, 252]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    model = AMR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, map_location='cpu')
    return model, model_cfg
