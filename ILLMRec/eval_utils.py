
import os
import torch
import warnings
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig
from ILLMRec.model.ILLMRec import ILLMRecLlamaConfig
from ILLMRec.model import *


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def prepare_config_for_eval(config: PretrainedConfig, device, kwargs: dict):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(f"Invalid configuration! Cannot find vision_tower in config:\n{config}")


class EmptyConfig(PretrainedConfig):
    def __init__(self):
        pass



def is_mm_model(model_path):
    """
    Check if the model at the given path is a visual language model.

    Args:
        model_path (str): The path to the model.
    Returns:
        bool: True if the model is an MM model, False otherwise.
    """
    config = ILLMRecLlamaConfig.from_pretrained(model_path)
    #config = AutoConfig.from_pretrained(model_path)
    architectures = config.architectures
    for architecture in architectures:
        if "illmrec" in architecture.lower():
            return True
    return False


def load_pretrained_model(
    model_path,
    model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    args=None,
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    config = AutoConfig.from_pretrained(model_path)
    config.resume_path = model_path
    prepare_config_for_eval(config, device_map, kwargs)
    

    config.data_args = EmptyConfig()
    for k, v in config.data_cfg.items():
        setattr(config.data_args, k, v )
    config.data_cfg = config.data_args
    config.rec_args = EmptyConfig()
    for k, v in config.rec_cfg.items():
        setattr(config.rec_args, k, v )
    config.rec_cfg = config.rec_args

    config.llm_args = EmptyConfig()
    for k, v in config.llm_cfg.items():
        setattr(config.llm_args, k, v )
    config.llm_cfg = config.llm_args

    config.train_args = EmptyConfig()
    for k, v in config.train_cfg.items():
        setattr(config.train_args, k, v )
    config.train_cfg = config.train_args
    
    config.llm_cfg._name_or_path = args.llm_name
    config.rec_cfg._name_or_path = args.model_path + "/rec_module"
    
    model = ILLMRecLlamaModel(
        config=config,
        low_cpu_mem_usage=True,
        **kwargs
    )

    tokenizer = model.tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        f"{model_path}/llm", 
        model_max_length=config.llm_cfg.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )
    model.tokenizer = tokenizer
    model.eval()
    image_processor = None
    if is_mm_model(model_path):
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor
    config.rec_cfg
    
    model.rec_module = model.rec_module.from_pretrained(config.rec_cfg._name_or_path)
    if os.path.isfile(os.path.join(model_path, "llm_parameter/token_embed.pth")):
        token_embed = torch.load(os.path.join(model_path, "llm_parameter/token_embed.pth"))
        model.llm.model.embed_tokens.weight.data = token_embed.cpu() if token_embed.is_cuda else token_embed
        print(f"Load Pretrained Token Embedding from {model_path}")

    
    for k, v in model.named_parameters():
        v.requires_grad = False

    return tokenizer, model, image_processor, config


