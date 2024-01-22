from models_dim import DiM_models

def create_model(config):
    return DiM_models[config.model_type](
        img_resolution=config.image_size // 8,
        in_channels=config.num_in_channels,
        label_dropout=config.label_dropout,
        num_classes=config.num_classes,
    )