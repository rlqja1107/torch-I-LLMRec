import os
import json
import torch
from ILLMRec.dataset import load_dataset
from ILLMRec.trainer import ILLMRecTrainer
from ILLMRec.model.utils import build_model
from ILLMRec.model.SASRec.model import SASRec
from transformers import HfArgumentParser, set_seed
from ILLMRec.args import PretrainedMLLMArguments, TrainingArguments, DataArguments, RecArguments
if "WANDB_PROJECT" not in os.environ:
    # Default to WANDB project "VILA".
    os.environ["WANDB_PROJECT"] = "ILLMRec"


def train():
    parser = HfArgumentParser((PretrainedMLLMArguments, TrainingArguments, RecArguments, DataArguments))
    mllm_args, training_args, rec_args, data_args = parser.parse_args_into_dataclasses()
    login_huggingface(rec_args)
    n_item = len(json.load(open(f"{data_args.data_path}/meta_info.json", 'r')))
    rec_args.n_item = n_item
    set_seed(training_args.seed)
    
    # Call Pre-trained VILO model
    ILLMRec, tokenizer = build_model(mllm_args, training_args, data_args, rec_args)

    # Initialize Projector & Aggregator
    data_dict = load_dataset(data_args, rec_args, tokenizer, train=True)
    cf_model_load(ILLMRec, rec_args, data_dict)
    # Load CF model


    training_args.sample_lens = [len(data_dict['train_dataset'])]
    trainer = ILLMRecTrainer(
        model=ILLMRec, tokenizer=tokenizer, args=training_args,
        **data_dict
    )
    #MLLM4Rec.train()
    print("Save Steps:", training_args.save_steps)
    if rec_args.load_features:
        if hasattr(ILLMRec, "vision_tower"):
            del ILLMRec.vision_tower
    
    print("Output Dir: ", training_args.output_dir)
    if rec_args.resume_model != "":
        load_pretrained_model(ILLMRec, rec_args.resume_model, rec_args, training_args)
    
    trainer.train(resume_from_checkpoint=False)



def login_huggingface(rec_args):
    from huggingface_hub import login
    # https://huggingface.co/meta-llama/Llama-2-7b-hf
    login(token=rec_args.hugging_token_id) # Need to fill the token id (LLaMA)
    print("Login Success")


def cf_model_load(model, rec_args, data_dict):
    usernum = len(data_dict['train_dataset'])
    itemnum = len(data_dict['train_dataset'].meta_data)
    rec_args.device = model.device
    model.recsys_model = SASRec(usernum, itemnum, rec_args)
    model.recsys_model.load_state_dict(torch.load(rec_args.recsys_model_path))
    for k, v in model.recsys_model.named_parameters():
        v.requires_grad = False
    model.recsys_model.eval()


def load_pretrained_model(MLLM4Rec, resume_path, rec_args, training_args):
    print(f"Pretrained Model Path: {resume_path}")
    load_name = ["LLM"]
    MLLM4Rec.train()
    MLLM4Rec.llm.requires_grad_(False)
    MLLM4Rec.llm.eval()
    for p in MLLM4Rec.get_input_embeddings().parameters():
        p.requires_grad = rec_args.train_token_emb
        
    for p in MLLM4Rec.get_output_embeddings().parameters():
        p.requires_grad = rec_args.train_llm_head
        
    if os.path.exists(os.path.join(resume_path, "proj4rec")):
        MLLM4Rec.proj4hs.load_state_dict(torch.load(os.path.join(resume_path, "proj4rec/model_proj4hs.pth")))
        MLLM4Rec.proj4hs.to(torch.bfloat16)
        load_name.append("proj4rec")
        
        MLLM4Rec.proj4modal.load_state_dict(torch.load(os.path.join(resume_path, "proj4rec/model_proj4modal.pth")))
        MLLM4Rec.proj4modal.to(torch.bfloat16)
        load_name.append("proj4modal")
                    
                    
    for name in  ["visual_modality"]:
        if os.path.exists(os.path.join(resume_path, name)):
            if name == 'collaborative_modality':
                getattr(MLLM4Rec, f"{name}_model").load_state_dict(torch.load(os.path.join(resume_path, f"{name}/model.pth")), strict=False)
            else:
                getattr(MLLM4Rec, f"{name}_model").load_state_dict(torch.load(os.path.join(resume_path, f"{name}/model.pth")), strict=True)
            getattr(MLLM4Rec, f"{name}_model").to(torch.bfloat16)
            load_name.append(name)
            if rec_args.finetune_interface:
                for k, v in getattr(MLLM4Rec, f"{name}_model").named_parameters():
                    v.requires_grad = True
                getattr(MLLM4Rec, f"{name}_model").train()
            else:
                for k, v in getattr(MLLM4Rec, f"{name}_model").named_parameters():
                    v.requires_grad = False
                getattr(MLLM4Rec, f"{name}_model").eval()                    
    
    # LoRa load
    if os.path.exists(os.path.join(resume_path, "llm_parameter")):
        if os.path.exists(os.path.join(resume_path, "llm_parameter/adapter.pth")):
            adapter_module = torch.load(os.path.join(resume_path, "llm_parameter/adapter.pth"))

            state_dict = MLLM4Rec.base_model.model.state_dict()
            cnt = 0 
            for k, v in state_dict.items():
                
               if 'default' in k: 
                   name = k.replace("default.", "")
               else:
                   name = k
               if name in adapter_module:
                   state_dict[k] = adapter_module[name]
                   cnt += 1
            MLLM4Rec.base_model.model.load_state_dict(state_dict)
            if not training_args.lora_freeze:
                for k, v in MLLM4Rec.llm.model.layers.named_parameters():
                    if 'lora' in k:
                        v.requires_grad = True
                MLLM4Rec.llm.train()
            print(f"# Loaded LoRa: {cnt}")
        if os.path.exists(os.path.join(resume_path, "llm_parameter/token_embed.pth")):
            token_embed = torch.load(os.path.join(resume_path, "llm_parameter/token_embed.pth"))
            MLLM4Rec.llm.model.embed_tokens.weight.data = token_embed.cpu() if token_embed.is_cuda else token_embed
            
    print(f"Loaded Module: {', '.join(load_name)}")