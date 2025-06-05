from .vit import vith


def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vith':
        return vith(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
