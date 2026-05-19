import os
import sys
import ast
import torch
import pickle
import argparse
import datetime
import numpy as np
sys.path.append('.')
from time import time
import torch.nn as nn
from tqdm import tqdm
torch.set_num_threads(4)
from copy import deepcopy
from transformers import set_seed
from torch.utils.data import DataLoader
from ILLMRec.utils import setup_logging
from ILLMRec.dataset import load_dataset
from transformers import LogitsProcessorList 
from ILLMRec.model.SASRec.model import SASRec
from ILLMRec.model import conversation as conversation_lib
from torch.utils.data.distributed import DistributedSampler
from ILLMRec.dataset import DataCollatorForSupervisedDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from ILLMRec.eval_utils import disable_torch_init, get_model_name_from_path, load_pretrained_model

MATCH_PREFERECE = {0: 'v', 1: 'c', 2: 't'}


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def name_setting(args):
    name_text = f"{args.model_path}/result_{args.n_negative_item}_{args.rec_cos}_{args.constrained_decode_type}"
    if args.soft_prediction:
        name_text += "_soft"
    
    if args.average:
        name_text += "_avg"
    
    if args.inference_normalize:
        name_text += "_normalize"
    else:
        name_text += "_no_normalize"
    return name_text + ".txt"




def load_resume_model(config, model, tokenizer, model_path):
    resume_path = config.rec_cfg.resume_model
    final_load_path = resume_path if resume_path and hasattr(config.rec_cfg, "adapter_merge") and config.rec_cfg.adapter_merge else model_path
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    lora_config = LoraConfig(r=64, 
                                lora_alpha=16, 
                                target_modules=['q_proj', 'v_proj'], 
                                lora_dropout=0.05,
                                bias='none',
                                task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_config)
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    model.base_model.model.pad_token_id = tokenizer.pad_token_id
    adapter_module = torch.load(os.path.join(final_load_path, "llm_parameter/adapter.pth"))
    state_dict = model.base_model.model.state_dict()
    cnt = 0 
    for k, v in state_dict.items():
        
        if 'default' in k: 
            name = k.replace("default.", "")
        else:
            name = k
        if name in adapter_module:
            state_dict[k] = adapter_module[name]
            cnt += 1
    model.base_model.model.load_state_dict(state_dict)
    model.merge_and_unload()
    print(f"LoRa Load Success from: {final_load_path} // cnt: {cnt}")
    
    if resume_path:
        token_embed = torch.load(os.path.join(resume_path, "llm_parameter/token_embed.pth"))
        model.llm.model.embed_tokens.weight.data = token_embed.cpu() if token_embed.is_cuda else token_embed
        print(f"Load Pretrained Token Embedding")


def open_neg_item(file_path):
    neg_item_dict = {}
    with open(file_path, 'r') as f:
        for i in f.readlines():
            i = i.split(" ")
            neg_item_dict[i[0]] = list(map(int, i[1:][:]))
    return neg_item_dict


def cf_model_load(model, rec_args, data_dict):
    usernum = len(data_dict['eval_dataset'])
    itemnum = len(data_dict['eval_dataset'].meta_data)
    rec_args.device = model.device
    model.recsys_model = SASRec(usernum, itemnum, rec_args)
    model.recsys_model.load_state_dict(torch.load(rec_args.recsys_model_path))
    for k, v in model.recsys_model.named_parameters():
        v.requires_grad = False
    model.recsys_model.eval()


def eval_model(args):
    args.device = 'cuda'
    # Model
    set_seed(333)
       
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, config = load_pretrained_model(model_path, model_name, args.model_base, args=args, device_map=args.device)
    
    logger_writer = setup_logging(f"{args.model_path}/result.txt", print_log=False)

    config.data_cfg.image_processor = image_processor
    data_dict = load_dataset(config.data_cfg, config.rec_cfg, tokenizer, train=False)
    cf_model_load(model,config.rec_cfg, data_dict)
    
    if config.rec_cfg.load_features:
        if hasattr(model, "vision_tower"):
            del model.vision_tower

    model = model.to(args.device)
    model = model.to(torch.float16)
    model.eval()
    generation_kwargs = {}
    collate_fn = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=config.data_cfg, train=False, generation_kwargs=generation_kwargs)
    eval_data_loader = DataLoader(data_dict['eval_dataset'], batch_size=args.eval_batch_size, pin_memory=True, drop_last=False, collate_fn=collate_fn.__call__) 

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]
    logger_writer.info(f"Start Time: {str(datetime.datetime.now())}")
    start = time()
    visual_features_all = torch.stack([torch.load(os.path.join(config.data_cfg.data_path, "img_features", f"{i}.pth")) for i in range(len(eval_data_loader.dataset.meta_data))]).to(model.device).to(torch.float16)
    textual_features_all =  torch.stack([torch.load(os.path.join(config.data_cfg.data_path, f"txt_features", f"{i}.pth")) for i in range(len(eval_data_loader.dataset.meta_data))]).to(model.device).to(torch.float16)
    
    negative_item = open_neg_item(f"{config.data_cfg.data_path}/neg_item_set.txt")
    
    for j, input_dict in tqdm(enumerate(eval_data_loader), total=len(eval_data_loader.dataset)//args.eval_batch_size):
        seq_img = None if len(input_dict['seq_img'][0]) ==0 else input_dict['seq_img'][0]
        with torch.inference_mode():
            _, llm_guided_user_representation = model(IRE_input_ids=input_dict['IRE_input_ids'], 
                        seq_img=seq_img, 
                        seq_id=input_dict['seq_id'],
                        IRE_attention_mask = input_dict['IRE_input_ids'].ne(tokenizer.pad_token_id),
                        generation_kwargs = generation_kwargs,
                        train=False
                        )
        for i in range(llm_guided_user_representation.shape[0]):
            target_id = input_dict['target_id'][i]
            neg_item_idx = negative_item[input_dict['user_name'][i]][:args.n_negative_item]
            neg_item_idx.append(target_id)
            rank = compute_rank(model, llm_guided_user_representation[i].unsqueeze(0).unsqueeze(0), neg_item_idx, visual_features=visual_features_all, textual_features=textual_features_all)
            logger_writer.info(f"Target: {target_id} // Rank: {rank}")

        if j % 500 == 0:
            print(f"Time: {time()-start}")
            
    end = time()
    logger_writer.info(f"End Time: {str(datetime.datetime.now())}")
    logger_writer.info(f"Total Time: {str(datetime.timedelta(seconds=end-start))}")

def compute_rank(model, llm_guided_user_representation, neg_item_idx, visual_features, textual_features):
    # Visual feature
    llm_guided_user_representation_v = model.rec_module(llm_guided_user_representation, 'img', u_i='user')[0]
    vis_feature = model.rec_module(visual_features[neg_item_idx], 'img', u_i='item')
    predictions = -torch.matmul(llm_guided_user_representation_v, vis_feature.T)
    v_predictions = predictions[0]
    
    # CF feature
    llm_guided_user_representation_cf = model.rec_module(llm_guided_user_representation, 'cf', u_i='user')[0]
    cf_feature = model.recsys_model.return_item_embed(deepcopy(np.array(neg_item_idx)), llm_guided_user_representation.device)
    cf_feature = model.rec_module(cf_feature, 'cf', u_i='item')
    cf_predictions = -torch.matmul(llm_guided_user_representation_cf, cf_feature.T)
    cf_predictions = cf_predictions[0]
    
    # Textual feature
    llm_guided_user_representation_txt = model.rec_module(llm_guided_user_representation, 'text', u_i='user')[0]
    txt_feature = model.rec_module(textual_features[neg_item_idx], 'text', u_i='item')
    txt_predictions = -torch.matmul(llm_guided_user_representation_txt, txt_feature.T)
    txt_predictions = txt_predictions[0]
    
    predictions = 1/3 * v_predictions + 1/3 * cf_predictions + 1/3 * txt_predictions
    # Only Part
    rank = predictions.argsort().argsort()[-1].item()
    return rank



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./output/Art/checkpoint-2")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="princeton-nlp/Sheared-LLaMA-2.7B")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--n_negative_item", type=int, default=100)


    args = parser.parse_args()
    print(f"Model Path: {args.model_path}, Batch Size: {args.eval_batch_size}")
    eval_model(args)
