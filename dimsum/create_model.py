from models_dim import DiM_models
from models_dit import DiT_models


def create_model(config):
    if "DiM" in config.model:
        return DiM_models[config.model](
            img_resolution=config.image_size // 8,
            in_channels=config.num_in_channels,
            label_dropout=config.label_dropout,
            num_classes=config.num_classes,
            gated_linear_unit=config.gated_linear_unit,
            routing_mode=config.routing_mode,
            num_moe_experts=config.num_moe_experts,
            is_moe=config.is_moe,
            learn_sigma=config.learn_sigma,
            scan_type=config.bimamba_type,
            pe_type=config.pe_type,
            block_type=config.block_type,
            cond_mamba=config.cond_mamba,
            scanning_continuity=config.scanning_continuity,
            enable_fourier_layers=config.enable_fourier_layers,
            drop_path=config.drop_path,
            rms_norm=config.rms_norm,
            fused_add_norm=config.fused_add_norm,
            learnable_pe=config.learnable_pe,
            use_final_norm=config.use_final_norm,
            use_attn_every_k_layers=config.use_attn_every_k_layers,
            use_gated_mlp=not config.not_use_gated_mlp,
        )
    elif "DiT" in config.model:
        return DiT_models[config.model](
            img_resolution=config.image_size // 8,
            in_channels=config.num_in_channels,
            label_dropout=config.label_dropout,
            num_classes=config.num_classes,
            learn_sigma=config.learn_sigma,
        )
