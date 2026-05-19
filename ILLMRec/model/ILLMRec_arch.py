#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import pickle
import warnings
from abc import ABC
import os, os.path as osp
from collections import OrderedDict
import torch
from transformers import (
    AutoConfig,
    PreTrainedModel,
)


from ILLMRec.model.language_model.builder import build_llm_and_tokenizer
from ILLMRec.model.multimodal_encoder.builder import build_vision_tower
from ILLMRec.model.learnable_module.builder import build_learnable_rec_module
from ILLMRec.model.configuration import ILLMRecConfig
from transformers.modeling_utils import ContextManagers, no_init_weights



    

## TODO decide whether should we use metaclass
class ILLMRecMetaModel(ABC):
    def init_vlm(self, config: PreTrainedModel = None, rec_cfg = None, *args, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if hasattr(self, "llm") or hasattr(self, "vision_tower"):
            # already initialized, skipped
            return 
        
        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        from ILLMRec.model.utils import get_model_config
        cfgs = get_model_config(config)
        if len(cfgs) == 2:
            llm_cfg, vision_tower_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `vision_tower_cfg` not found in the config.")
        if hasattr(config.llm_cfg, "_name_or_path") and len(config.llm_cfg._name_or_path.split("/")) <= 2:
            llm_cfg = config.llm_cfg._name_or_path
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.rec_module = build_learnable_rec_module(rec_cfg, config)
        
        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."



    @classmethod
    def load_from_config(cls, model_path_or_config, *args, **kwargs):
        pass
    
    ## FIXME we will use this function to load model in the future
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        kwargs.pop("config", None)

        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config)
        elif isinstance(model_path_or_config, ILLMRecConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, ILLMRecConfig)}")

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        from ILLMRec.model.utils import get_model_config
        cfgs = get_model_config(config)
        if len(cfgs) == 2:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained")
        with ContextManagers([no_init_weights(_enable=True),]):
            vlm = cls(config, *args, **kwargs)
        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained finish")
        
        if hasattr(vlm, "llm") or hasattr(vlm, "vision_tower")  or hasattr(vlm, "mm_projector"):
            if vlm.is_loaded:
                return vlm
        
        vlm.llm, vlm.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        vlm.vision_tower = build_vision_tower(vision_tower_cfg, config)
        assert (
            vlm.llm is not None or vlm.vision_tower is not None or vlm.mm_projector is not None
        ), "At least one of the components must be instantiated."
        return vlm
    
    ## FIXME we will use this function to save the model in the future
    def save_pretrained(self, output_dir, state_dict=None, safe_serialization=True):
        if state_dict is None:
            state_dict = self.state_dict()
        
        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            # Load Code
            os.makedirs(os.path.join(output_dir, "llm_parameter"), exist_ok=True)
            if self.rec_args.train_token_emb:
                token_embed_weight = self.llm.model.embed_tokens.weight.data
                torch.save(token_embed_weight, os.path.join(os.path.join(output_dir, "llm_parameter"), "token_embed.pth"))

            self.config.llm_cfg = self.llm.config

        if self.get_vision_tower() and "radio" not in self.get_vision_tower().__class__.__name__.lower():
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vision_tower.config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            self.vision_tower.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.vision_tower.image_processor.save_pretrained(os.path.join(output_dir, "vision_tower"))
            self.config.vision_tower_cfg = self.vision_tower.config
            if hasattr(self.config.vision_tower_cfg, 'auto_map'):
                delattr(self.config.vision_tower_cfg, 'auto_map')
            
        if self.get_rec_module():
            print(f"saving visual_modality to {osp.join(output_dir, 'rec_module')}")
            self.rec_module.config._name_or_path = osp.join(output_dir, "rec_module")
            os.makedirs(os.path.join(output_dir, "rec_module"), exist_ok=True)
            self.rec_module.save_pretrained(os.path.join(output_dir, "rec_module"))


        if hasattr(self.config, "data_cfg"):
            if not isinstance(self.config.data_cfg, dict):
                data_cfg_dict = {}
                for k, v in self.config.data_cfg.__dict__.items():
                    if k == 'image_processor': continue
                    data_cfg_dict[k] = v
                self.config.data_cfg = data_cfg_dict
        if hasattr(self.config, "rec_cfg"):
            if not isinstance(self.config.rec_cfg, dict):
                rec_cfg_dict = {}
                for k, v in self.config.rec_cfg.__dict__.items():
                    rec_cfg_dict[k] = v
                self.config.rec_cfg = rec_cfg_dict
            if "device" in self.config.rec_cfg:
                del self.config.rec_cfg['device']
        if hasattr(self.config, "train_cfg"):
            if not isinstance(self.config.train_cfg, dict):
                train_cfg_dict = {}
                for k, v in self.config.train_cfg.__dict__.items():
                    if isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                        train_cfg_dict[k] = v
                self.config.train_cfg = train_cfg_dict
      
        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)
   

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head


    def get_textual_modality(self):
        return getattr(self, "textual_modality_model", None)

    def get_rec_module(self):
        return getattr(self, "rec_module", None)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector

    def post_config(self):
        self.training = self.get_llm().training
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config


    def freezed_module_patch(self):
        '''
        Huggingface will call model.train() at each training_step. To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        '''
        if self.training:
            if self.get_vision_tower() and not getattr(self.config, "tune_vision_tower", False):
                self.get_vision_tower().eval()
            if self.get_mm_projector() and not getattr(self.config, "tune_mm_projector", False):
                self.get_mm_projector().eval()
    
    def encode_images(self, images):
        image_features, cls_feature = self.get_vision_tower()(images)
        #image_features = self.get_mm_projector()(image_features)
        #cls_feature = self.get_mm_projector()(cls_feature.unsqueeze(1))
        return image_features, cls_feature
    
    ## @yunhao: is there a better way to handle function call and attributes for llm?
    ## support beam search
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)

    

class ILLMRecMetaForCausalLM(ABC):
    """This class is originally implemented by the LLaVA team and
    modified by Haotian Tang and Jason Lu based on Ji Lin's implementation
    to support multiple images and input packing."""




    def initialize_vision_tokenizer(self):
        for p in self.get_input_embeddings().parameters():
            p.requires_grad = self.rec_args.train_token_emb
            
        for p in self.get_output_embeddings().parameters():
            p.requires_grad = False



