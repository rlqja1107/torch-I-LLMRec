import torch
from ILLMRec.model.learnable_module.projector import ProjectorConfig, Projector
from transformers import AutoConfig, PretrainedConfig

def build_learnable_rec_module(rec_cfg: PretrainedConfig, config):
    if not hasattr(rec_cfg, "visual_encoder"):
        vt_cfg = (getattr(config, "vision_tower_cfg", "") or "").lower()
        if "siglip" in vt_cfg:
            rec_cfg.visual_encoder = "siglip"
        elif "clip" in vt_cfg:
            rec_cfg.visual_encoder = "clip"
        else:
            raise ValueError(
                f"Cannot infer rec_cfg.visual_encoder from vision_tower_cfg={vt_cfg!r}"
            )
    rec_config = ProjectorConfig(rec_cfg)
    # All the learnable parameters are in Projector
    rec_module = Projector(rec_config).to(eval(config.model_dtype))
    return rec_module
    