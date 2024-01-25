from models_dim import DiM_models
from models_dit import DiT_models

def create_model(config):
    if "DiM" in config.model:
        return DiM_models[config.model](
            img_resolution=config.image_size // 8,
            in_channels=config.num_in_channels,
            label_dropout=config.label_dropout,
            num_classes=config.num_classes,
        )
    elif "DiT" in config.model: 
        return DiT_models[config.model](
            img_resolution=config.image_size // 8,
            in_channels=config.num_in_channels,
            label_dropout=config.label_dropout,
            num_classes=config.num_classes,
        )

