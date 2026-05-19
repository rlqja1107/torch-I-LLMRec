# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/
import torch
import transformers
from typing import Dict
import os, os.path as osp
from dataclasses import dataclass
from transformers import  PretrainedConfig
from huggingface_hub.utils import HFValidationError
from huggingface_hub import snapshot_download, repo_exists
from transformers import PretrainedConfig, PreTrainedModel
from ILLMRec.model import conversation as conversation_lib
from ILLMRec.model.ILLMRec import ILLMRecLlamaModel, ILLMRecLlamaConfig


def get_model_config(config):
    default_keys = ["llm_cfg", "vision_tower_cfg"]
    
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path 
        
    # download from huggingface
    if root_path is not None and not osp.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError as e:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)

    return_list = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_list.append(os.path.join(root_path, key[:-4]))
            except:
                raise ValueError(f"Cannot find resume path in config for {key}!")
        elif isinstance(cfg, PretrainedConfig):
            return_list.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_list.append(cfg)
    return return_list


def prepare_config_for_training(
    config: PretrainedConfig, model_args: dataclass, training_args: dataclass, rec_args: dataclass, data_args: dataclass) -> None:
    assert model_args.vision_tower is not None, "requires vision tower"
    ## set module configurations
    if getattr(config, "llm_cfg", None) is None:
        config.llm_cfg = model_args.model_name_or_path
    if getattr(config, "vision_tower_cfg", None) is None:
        config.vision_tower_cfg = model_args.vision_tower
    if getattr(config, "mm_projector_cfg", None) is None:
        config.mm_projector_cfg = model_args.mm_projector
    
    config.rec_cfg = rec_args
    config.data_cfg = data_args
    config.train_cfg = training_args
    # data argument
    
    ## set default dtype
    config.model_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    config.model_dtype = config.model_dtype.__str__()
    ## set tuning modules
    config.tune_language_model = training_args.tune_language_model
    config.tune_vision_tower = training_args.tune_vision_tower
    config.tune_mm_projector = training_args.tune_mm_projector
    ## set data args
    config.image_aspect_ratio = "resize"
    ## extra vision tower configuration
    if getattr(config, "vision_tower_cfg", None) is not None:
        config.mm_vision_select_layer = model_args.mm_vision_select_layer
        config.mm_vision_select_feature = model_args.mm_vision_select_feature
        ## vision tower configurations
        config.vision_resolution = model_args.vision_resolution
        config.interpolate_mode = model_args.interpolate_mode
        config.drop_path_rate = model_args.drop_path_rate
        config.s2 = model_args.s2
        config.s2_scales = model_args.s2_scales
        config.s2_max_split_size = model_args.s2_max_split_size



def vision_resolution_elevation(model: PreTrainedModel, config: PretrainedConfig):
    vision_tower = model.get_vision_tower()
    if (
        vision_tower is not None
        and "radio" not in vision_tower.__class__.__name__.lower()
    ):
        vision_tower._maybe_resize_pos_embeds(
            model=vision_tower.vision_tower,
            image_processor=vision_tower.image_processor,
            resolution=getattr(config, "vision_resolution", -1),
            interpolate_mode=getattr(config, "interpolate_mode", "linear"),
        )


def mprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if rank == 0:
            return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
        else:
            return
    else:
        return print(*args, **kwargs)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    #model.resize_token_embeddings(len(tokenizer))


def add_extra_tokens(tokenizer, model):
    tokenizer.add_tokens(['<i_vis>'])
    model.get_llm().resize_token_embeddings(len(tokenizer))
    


def build_model(mllm_args, training_args, data_args, rec_args):
    # Load LLM, Visual Tower
    model_cls = ILLMRecLlamaModel
    config = ILLMRecLlamaConfig.from_pretrained(
        mllm_args.model_name_or_path,
        resume=False
    )
    if getattr(config, "resume_path", None) is not None:
        config.resume_path = mllm_args.model_name_or_path
        
    ## extra configurations
    prepare_config_for_training(config, mllm_args, training_args, rec_args, data_args)
    
    bnb_model_from_pretrained_args = {}
    model = model_cls(
        config=config,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args,
    )
    vision_resolution_elevation(model, config)
    
    model.llm.config.use_cache = False
    model.get_llm().requires_grad_(training_args.tune_language_model)
    model.get_llm().eval()
        
    mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")
    if model.get_vision_tower():
        model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
        model.get_vision_tower().eval()

    def need_to_modify_do_sample(generation_config):
        if generation_config.do_sample is False:
            if (
                generation_config.temperature is not None
                and generation_config.temperature != 1.0
            ):
                return True
            if generation_config.top_p is not None and generation_config.top_p != 1.0:
                return True
        return False
    
    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    if training_args.gradient_checkpointing:
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model.llm,
        )
    if mllm_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            mllm_args.version
        ]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]
    
    model.llm.pad_token_id = tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = tokenizer.model_max_length    

    vision_tower = model.get_vision_tower()
    data_args.image_processor = vision_tower.image_processor

    #model.config.num_video_frames = data_args.num_video_frames
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
        mllm_args.mm_use_im_start_end
    )
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = mllm_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = mllm_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer()
    
    # Placeholder of visual features
    tokenizer.add_tokens(['<i_vis>'])
    model.get_llm().resize_token_embeddings(len(tokenizer))

    return model, tokenizer



    
