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

# This file is modified from https://github.com/haotian-liu/LLaVA/

import os
import torch
import warnings
import numpy as np
from copy import deepcopy
from typing import List, Optional, Tuple, Union
warnings.filterwarnings('ignore')
from .configuration import ILLMRecConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .ILLMRec_arch import ILLMRecMetaModel, ILLMRecMetaForCausalLM
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)



class ILLMRecLlamaConfig(ILLMRecConfig):
    model_type = "illmrec_llama"


## FIXME we will follow the convention to add a new class for CausalLM in the future
class ILLMRecLlamaModel(ILLMRecMetaModel, ILLMRecMetaForCausalLM, PreTrainedModel):
    config_class = ILLMRecLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    
    
    def __init__(self, config: ILLMRecLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.data_path = config.data_cfg.data_path
        self.rec_args = config.rec_cfg
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.print_first = True # For print
        return self.init_vlm(config=config, rec_cfg=self.rec_args, *args, **kwargs)
    
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(pretrained_model_name_or_path, 
                *model_args, config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, force_download=force_download, local_files_only=local_files_only, token=token, 
                revision=revision, use_safetensors=use_safetensors, **kwargs
            )
        return super(ILLMRecLlamaModel).from_pretrained(pretrained_model_name_or_path, 
            *model_args, config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, force_download=force_download, local_files_only=local_files_only, token=token, 
            revision=revision, use_safetensors=use_safetensors, **kwargs)    

    
    def visual_feature_to_LLM(self, i, input_ids, inputs_embeds, seq_id, vis_feature):
        n_interaction = len(seq_id)
        item_img_idx = torch.where(input_ids == self.tokenizer("<i_vis>")['input_ids'][-1])[0]
        if len(item_img_idx) > 0:
            inputs_embeds[i][item_img_idx[:n_interaction]] = self.rec_module.adaptor(vis_feature[:n_interaction])


    def encode_visual_feature(self, images):
        concat_imgs = torch.cat([torch.cat([images[i]]) for i in range(len(images))])
        if concat_imgs.shape[0] > 32:
            vis_feature_list = []
            idx = 0
            while idx < concat_imgs.shape[0]:
                if idx+32 > concat_imgs.shape[0]:
                    _, vis_feature = self.encode_images(concat_imgs[idx:concat_imgs.shape[0]])
                else:
                    _, vis_feature = self.encode_images(concat_imgs[idx:idx+32])
                idx += 32
                vis_feature_list.append(vis_feature)
            vis_feature = torch.cat(vis_feature_list)
        else:
            _, vis_feature = self.encode_images(concat_imgs)
        vis_feature = vis_feature / vis_feature.norm(p=2, dim=-1, keepdim=True)
        vis_feature = vis_feature.split([images[i].shape[0] + self.rec_args.max_candidate + 1 for i in range(len(images))])
        return vis_feature
    
    
    def encode_textual_feature(self, description_list, seq_id):
        new_text_list = []
        idx = 0
        for i in range(len(seq_id)):
            new_text_list.extend(description_list[i])
            idx += len(seq_id[i])
            
        encoded_input = self.textual_modality_model.tokenizer(new_text_list, truncation=True, padding=True, return_tensors='pt')
        encoded_input['input_ids'] = encoded_input['input_ids'].to(self.device)
        encoded_input['attention_mask'] = encoded_input['attention_mask'].to(self.device)
        encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(self.device)
        with torch.no_grad():
            text_output = self.textual_modality_model.model(**encoded_input)
        text_features = text_output['pooler_output'].split([seq_id[i].shape[0] for i in range(len(seq_id))])
        return text_features


    def freeze_parameters(self):
        self.llm.eval()
               
    
    def forward(
        self,
        IRE_input_ids: torch.LongTensor = None,
        seq_img: Optional[torch.FloatTensor] = None,
        IRE_attention_mask: Optional[torch.Tensor] = None,
        seq_id: Optional[List] = None,
        target_id: Optional[List] = None,
        ILA_input_ids: torch.LongTensor = None,
        ILA_labels: torch.LongTensor = None,
        user_name: Optional[List] = None,
        ILA_attention_mask: Optional[torch.Tensor] = None,
        generation_kwargs = None,
        train=True
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #self.freeze_parameters()
        if not train:
            return self.generate(IRE_input_ids=IRE_input_ids, 
                                    seq_img=seq_img, 
                                    seq_id=seq_id,
                                    IRE_attention_mask=IRE_attention_mask,
                                    generation_kwargs=generation_kwargs,
                                    )
        # Load Pretrained Visual Feature
        if self.rec_args.load_features:
            vis_feature = []
            for s in seq_id:
                per_user_vis_feature = torch.stack([torch.load(os.path.join(self.data_path, "img_features", f"{i}.pth")) for i in s]).to(self.device)
                vis_feature.append(per_user_vis_feature)
        else:
            vis_feature = self.encode_visual_feature(seq_img)
        
        
        # ILA Module
        ILA_output = self.ILA_module(ILA_input_ids, seq_id, ILA_attention_mask, vis_feature, ILA_labels)
        
        # IRE Module
        IRE_output = self.IRE_Module(IRE_input_ids, seq_id, vis_feature, target_id)
        IRE_loss = IRE_output['loss']; ILA_loss = ILA_output['loss']
        IRE_output['loss'] += ILA_output['loss']
        if self.print_first:
            print(f"ILA Loss: {ILA_loss.item()}, IRE Loss: {IRE_loss.item()}")
            self.print_first = False
        return IRE_output


    def ILA_module(self, ILA_input_ids, seq_id, ILA_attention_mask, vis_feature, ILA_labels):
        ILA_input_embeds = self.llm.get_input_embeddings()(ILA_input_ids)
        ILA_position_list = []
        for i in range(len(seq_id)):
            ILA_position_list.append(torch.arange(ILA_attention_mask[i].sum(), device=ILA_input_embeds[i].device))
        ILA_position_list = torch.nn.utils.rnn.pad_sequence(ILA_position_list, batch_first=True, padding_value=-1)
        
        for i in range(len(seq_id)):
            self.visual_feature_to_LLM(i, ILA_input_ids[i], ILA_input_embeds, seq_id[i], vis_feature[i])

        ILA_output = self.llm.forward(
            input_ids=None,
            attention_mask=ILA_attention_mask,
            position_ids=ILA_position_list,
            past_key_values=None,
            inputs_embeds=ILA_input_embeds,
            labels=ILA_labels,
        )
        return ILA_output


    def IRE_Module(self, IRE_input_ids, seq_id, vis_feature, target_id):
        IRE_inputs_embeds = self.llm.get_input_embeddings()(IRE_input_ids)
        last_hidden_state_idx =  [torch.where(i == 0)[0][0].item() if len(torch.where(i == 0)[0]) > 0 else i.shape[0] for i in IRE_input_ids]
        # Concat
        input_embeds_concat_list = []
        attention_mask_cocat_list = []
        for i, idx in enumerate(last_hidden_state_idx):
           concat_input_embeds = torch.cat([IRE_inputs_embeds[i][:idx], self.rec_module.rec_token, IRE_inputs_embeds[i][idx:]])
           input_embeds_concat_list.append(concat_input_embeds)
           attn_mask = torch.cat([IRE_input_ids[i], torch.LongTensor([10]).to(IRE_inputs_embeds.device)]).ne(self.tokenizer.pad_token_id)
           attn_mask[idx] = True
           attention_mask_cocat_list.append(attn_mask)
        IRE_inputs_embeds = torch.stack(input_embeds_concat_list)
        IRE_attention_mask = torch.stack(attention_mask_cocat_list)
        
        
        for i in range(len(seq_id)):
            self.visual_feature_to_LLM(i, IRE_input_ids[i], IRE_inputs_embeds, seq_id[i], vis_feature[i])
                
        IRE_position_list = []
        
        for i in range(len(seq_id)):
            IRE_position_list.append(torch.arange(IRE_attention_mask[i].sum(), device=self.device))
        IRE_position_list = torch.nn.utils.rnn.pad_sequence(IRE_position_list, batch_first=True, padding_value=-1)

        IRE_output = self.llm.forward(
            input_ids=None,
            attention_mask=IRE_attention_mask,
            position_ids=IRE_position_list,
            past_key_values=None,
            inputs_embeds=IRE_inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        # Retrieval task - [REC] Token
        hidden_states = IRE_output['hidden_states'][-1] # Last hidden states
           
        # LLM-guided user representation
        llm_user_guided_rep = torch.stack([hidden_states[i, j:j+1, :] for i, j in enumerate(last_hidden_state_idx)], axis=0)
        llm_user_guided_rep = llm_user_guided_rep[:,0,:]
        IRE_loss = 0
        for f_t in ['img', 'cf', 'text']:
            llm_user_guided_proj = self.rec_module(llm_user_guided_rep, f_t, 'user')
            IRE_loss += self.compute_IRE_loss(target_id, llm_user_guided_proj, f_t=f_t, seq_id=seq_id)
        IRE_loss = (IRE_loss / 3)
        IRE_output['loss'] = IRE_loss
        return IRE_output


    def compute_IRE_loss(self, target_id, llm_user_guided_proj, f_t, seq_id):
        neg_item_list = []
        gt_label_list = []
        num_neg_item = 1
        n_batch = len(seq_id)
        for i in range(n_batch):
            # Random Negative Item Sampling
            neg_item = np.append(np.random.choice(np.delete(np.arange(self.rec_args.n_item), [target_id[i]]), num_neg_item, replace=False), target_id[i])
            gt_label = torch.LongTensor([num_neg_item]).to(llm_user_guided_proj.device)
            gt_label_list.append(gt_label)
            neg_item_list.append(neg_item)
        gt_label_list = torch.cat(gt_label_list)
        neg_item_list = np.concatenate(neg_item_list)
        pos_labels = torch.ones(int(neg_item_list.shape[0]/2), device = llm_user_guided_proj.device)
        neg_labels = torch.zeros(int(neg_item_list.shape[0]/2), device= llm_user_guided_proj.device)
        
        if 'cf' == f_t:
            item_feature = self.recsys_model.return_item_embed(deepcopy(neg_item_list), llm_user_guided_proj.device)
        elif 'text' == f_t:
            item_feature = torch.stack([torch.load(os.path.join(self.data_path, f"txt_features", f"{i}.pth")) for i in neg_item_list]).to(llm_user_guided_proj.device)
        elif 'img' == f_t:
            item_feature = torch.stack([torch.load(os.path.join(self.data_path, "img_features", f"{i}.pth")) for i in neg_item_list]).to(llm_user_guided_proj.device)
        else:
            assert False, "Please provide a feature type limited to only the cf, text and img"
            
        item_feature = self.rec_module(item_feature, f_t, 'item')
        
        pos_logit = (llm_user_guided_proj * item_feature[np.arange(1, neg_item_list.shape[0], 2),:]).sum(dim=-1)
        neg_logit = (llm_user_guided_proj * item_feature[np.arange(0, neg_item_list.shape[0], 2),:]).sum(dim=-1)
        IRE_loss = self.bce_criterion(pos_logit, pos_labels) + self.bce_criterion(neg_logit, neg_labels)
        
        return IRE_loss


 

    @torch.no_grad()
    def generate(
        self,
        IRE_input_ids: Optional[torch.FloatTensor] = None,
        seq_img: Optional[torch.FloatTensor] = None,
        IRE_attention_mask: Optional[torch.LongTensor] = None,
        seq_id = None,
        generation_kwargs = None,
    ):
        IRE_input_ids = IRE_input_ids.to(self.device)
        inputs_embeds = self.get_input_embeddings()(IRE_input_ids)
        inputs_embeds = self.refine_input_embeds(IRE_input_ids = IRE_input_ids,
                            seq_img = seq_img,
                            seq_id = seq_id,
                            inputs_embeds = inputs_embeds
                            )

        if IRE_attention_mask.device != inputs_embeds.device:
            IRE_attention_mask = IRE_attention_mask.to(inputs_embeds.device)


        last_hidden_state_idx =  [torch.where(i == 0)[0][0].item() if len(torch.where(i == 0)[0]) > 0 else i.shape[0] for i in IRE_input_ids]
        # Concat
        input_embeds_concat_list = []
        attention_mask_cocat_list = []
        for i, idx in enumerate(last_hidden_state_idx):
            concat_input_embeds = torch.cat([inputs_embeds[i][:idx], self.rec_module.rec_token, inputs_embeds[i][idx:]]) # Append Rec token
            input_embeds_concat_list.append(concat_input_embeds)
            attn_mask = torch.cat([IRE_input_ids[i], torch.LongTensor([10]).to(inputs_embeds.device)]).ne(self.tokenizer.pad_token_id)
            attn_mask[idx] = True
            attention_mask_cocat_list.append(attn_mask)
        inputs_embeds = torch.stack(input_embeds_concat_list)
        IRE_attention_mask = torch.stack(attention_mask_cocat_list)

        outputs = self.llm(inputs_embeds=inputs_embeds,
                            attention_mask=IRE_attention_mask,
                            output_attentions=True,
                            output_hidden_states=True,
                            )

        last_hidden_states = outputs['hidden_states'][-1] # Last hidden states
        llm_user_guided_rep = torch.stack([last_hidden_states[i, j:j+1, :] for i, j in enumerate(last_hidden_state_idx)], axis=0)
        llm_user_guided_rep = llm_user_guided_rep[:,0,:]

        return outputs, llm_user_guided_rep


    def refine_input_embeds(self, IRE_input_ids: Optional[torch.FloatTensor] = None,
        seq_img: Optional[torch.FloatTensor] = None,
        seq_id = None,
        inputs_embeds = None
        ):   
        if self.rec_args.load_features:
            vis_feature = []
            for s in seq_id:
                per_user_vis_feature = [torch.load(os.path.join(self.data_path, "img_features", f"{i}.pth")) for i in s]
                per_user_vis_feature = torch.stack(per_user_vis_feature).to(torch.float16).to(self.device) if len(per_user_vis_feature) > 0 else []
                vis_feature.append(per_user_vis_feature)
        else:
            seq_img = seq_img.to(self.device)
            vis_feature = self.encode_visual_feature(seq_img)
                
        for i in range(len(seq_id)):
            self.visual_feature_to_LLM(i, IRE_input_ids[i], inputs_embeds, seq_id[i], vis_feature[i])
        return inputs_embeds
    
    

AutoConfig.register("illmrec_llama", ILLMRecLlamaConfig)
AutoModel.register(ILLMRecLlamaConfig, ILLMRecLlamaModel)


class _LegacyLlavaLlamaConfig(ILLMRecLlamaConfig):
    model_type = "llava_llama"


AutoConfig.register("llava_llama", _LegacyLlavaLlamaConfig)
