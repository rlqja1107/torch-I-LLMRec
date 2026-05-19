# https://github.com/Efficient-Large-Model/VILA/blob/main/llava/train/args.py
import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PretrainedMLLMArguments:
    # output_dir: Optional[str] = field(default='result')
    version: Optional[str] = field(default="v1")
    model_name_or_path: Optional[str] = field(default="princeton-nlp/Sheared-LLaMA-2.7B")
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    mm_projector: Optional[str] = field(default="mlp_downsample")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)  # default to the last layer
    mm_vision_select_feature: Optional[str] = field(default="cls_patch")
    vision_resolution: Optional[int] = field(default=-1)
    interpolate_mode: Optional[str] = field(default="linear")
    drop_path_rate: Optional[float] = field(default=0.)
    s2: bool = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")
    s2_max_split_size: int = field(default=336)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    logging_steps: int = 5000
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    tune_vision_tower: bool = field(default=False)
    tune_language_model: bool = field(default=False)
    tune_mm_projector: bool = field(default=True)
    bf16: bool = field(default=True)
    num_train_epochs: Optional[float] = 70
    model_dtype: str = field(default="torch.bfloat16")
    gradient_checkpointing: bool = field(default=True)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    total_time_limit: int = field(
        default=-1, metadata={"help": "Timeout limit for this job (in minutes)."}
    )
    pre_terminate_time: int = field(
        default=10,
        metadata={
            "help": "Time to terminate the task inadvance (minutes), saveing checkpoints needs time."
        },
    )
    save_steps: int = 12000
    
@dataclass
class DataArguments:
    data_path: str = field(
        default="./dataset/Amazon_18/Art", metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "resize"
    eval_data_mixture: str = None
    vflan_no_system_prompt: bool = False
    downsample_video: bool = False
    data_type: str = "Art"


@dataclass
class RecArguments:
    recsys_model_path: str = "./dataset/Amazon_18/Art/SASRec_Art.pth"
    hidden_units: int = 50
    dropout_rate: float = 0.2
    maxlen: int = 50
    num_heads: int = 1
    num_blocks: int = 2
    max_interaction: int = field(default=10, metadata={"help": "Constraint of Max interaction"})
    load_features: bool = True
    proj_dim: int = 512 # If col => 50, others: 512
    llm_dim: int = 2560
    train_token_emb: bool = True
    visual_encoder: str = 'siglip'
    resume_model: str = ''
    hugging_token_id: str = ''