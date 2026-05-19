import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig
    )


v_dim = {'siglip': 1152, 'clip': 768}
modality_proj_dim = {'c': 50, 'v_siglip': 1152, 't': 768}


class ProjectorConfig(PretrainedConfig):
    model_type = "projector"
    def __init__(self,
                 rec_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        if rec_cfg:
            assert rec_cfg.visual_encoder in v_dim, "Error in the name of vision tower"
            self.visual_encoder = rec_cfg.visual_encoder
            self.llm_dim = rec_cfg.llm_dim
            self.proj_dim = rec_cfg.proj_dim
        

class Projector(PreTrainedModel):
    config_class = ProjectorConfig
    def __init__(self, config: ProjectorConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.adaptor = Adaptor(v_dim[config.visual_encoder], config.llm_dim)
        self.f_i_img = MLP(modality_proj_dim[f"v_{config.visual_encoder}"], config.proj_dim)
        self.f_i_cf = MLP(modality_proj_dim['c'], config.proj_dim)
        self.f_i_text = MLP(modality_proj_dim['t'], config.proj_dim)
        
        self.f_u_img = MLP(config.llm_dim, config.proj_dim)
        self.f_u_cf = MLP(config.llm_dim, config.proj_dim)
        self.f_u_text = MLP(config.llm_dim, config.proj_dim)
        self.rec_token = nn.Parameter(torch.randn(config.llm_dim).unsqueeze(0)) # [REC] token
        torch.nn.init.xavier_uniform(self.rec_token)

    def forward(self, x, f_t, u_i):
        if u_i == 'user':
            return eval(f"self.f_u_{f_t}")(x)
        elif u_i == 'item':
            return eval(f"self.f_i_{f_t}")(x)
        else:
            assert False, "Please provide a type limited to only the user and the item"

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim*2),
            nn.LayerNorm(out_dim*2),
            nn.ReLU(),
            nn.Linear(out_dim*2, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)


class Adaptor(nn.Module):
    def __init__(self, v_dim, llm_dim, intermediate_dim=512):
        super(Adaptor, self).__init__()
        self.proj_v2c = nn.Linear(v_dim, intermediate_dim)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(intermediate_dim, eps=1e-8)
        
        self.interface = nn.Linear(intermediate_dim, llm_dim, dtype=torch.bfloat16)
    
    def forward(self, x):
        text_feat = self.layernorm(self.proj_v2c(x))
        text_feat = self.activation(text_feat)
        return self.interface(text_feat)

AutoConfig.register("projector", ProjectorConfig)
AutoModel.register(ProjectorConfig, Projector)
