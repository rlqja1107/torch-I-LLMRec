import torch
from ILLMRec.model.learnable_module.projector import ProjectorConfig, Projector
from transformers import AutoConfig, PretrainedConfig

def build_learnable_rec_module(rec_cfg: PretrainedConfig, config):
    rec_config = ProjectorConfig(rec_cfg)
    # All the learnable parameters are in Projector 
    rec_module = Projector(rec_config).to(eval(config.model_dtype))
    return rec_module
    