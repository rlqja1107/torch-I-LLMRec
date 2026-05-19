import os
import PIL
import json
import copy
import torch
import numpy as np
import transformers
from typing import Dict, Sequence
from ILLMRec.model.constant import *
from PIL import ImageFile
from dataclasses import dataclass
from torch.utils.data import Dataset
from ILLMRec.args import DataArguments, RecArguments
from ILLMRec.utils import process_image, is_gemma_tokenizer
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000
from ILLMRec.model import conversation as conversation_lib



def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
    train: str = 'train'
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    if no_system_prompt:
        conv.system = ""
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    if train != 'test':
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
    else:
        for i, source in enumerate(sources):
            conv.messages = []
            for j, sentence in enumerate(source[:1]):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        if len(conversations) ==1:
            #conversations[0] = conversations[0][:-15]
            conversations[0] = conversations[0] + f"{conv.roles[1]}:"
    # Tokenize conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    if train != 'train':
        return dict(
        input_ids=input_ids,
        labels=targets,
    )
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                if i > 0 and not is_gemma_tokenizer(tokenizer):
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                if i > 0 and not is_gemma_tokenizer(tokenizer):
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}" f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    train: str = 'train',
    no_system_prompt: bool = False,  # only work for v1
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    #if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
    #    return preprocess_plain(sources, tokenizer)
    #if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
    #    return preprocess_llama_3(sources, tokenizer, has_image=has_image, no_system_prompt=False, train=train)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, no_system_prompt=False, train=train)
    assert True, "No Conversation style"

    
class LazyRecDataset(Dataset):
    """This class is implemented by Ji Lin and Haotian Tang."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        rec_args: RecArguments = None,
        split:str = 'train'
    ):
        super(LazyRecDataset, self).__init__()
        self.rec_args = rec_args
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = f"{data_path}/image"
        
        self.split = split
        if split == 'train':
            print("========= Loading Recommendation Dataset =========")
            self.user_data = json.load(open(f"{data_path}/train_users.json", 'r'))
        elif split == 'valid':
            self.user_data = json.load(open(f"{data_path}/eval_users.json", 'r'))
        else:
            self.user_data = json.load(open(f"{data_path}/test_users.json", 'r'))
        remove_key = [k  for k, v in self.user_data.items() if len(v) < 3]
        for r in remove_key:
            del self.user_data[r]
            
               
        self.user_idx2name = {i:k for i, k in enumerate(list(self.user_data.keys()))}
        self.meta_data = json.load(open(f"{data_path}/meta_info.json", 'r')) # idx: 0
        self.n_item = len(self.meta_data)
        # Hyperparameter
        self.max_length = rec_args.max_interaction
        self.print_first = True

        
    def get_meta_info(self, item_idx, key):
        item_info = self.meta_data[str(item_idx)][key]
        return item_info if item_info != '' else "Unknown"


    def __len__(self):
        return len(self.user_data)
    
        
    def prompt_per_item(self, item_id):
        title = self.get_meta_info(item_id, 'title')
        text = f"{{Title: {title}, Visual representation: <i_vis>}}, "
        return text
    
    
    def process_IRE_prompt(self, i):
        user_name = self.user_idx2name[i]
        user_seq = self.user_data[user_name]
        task_prompt = conversation_lib.task_prompt
        
        
        if self.split == 'train':
            if np.random.uniform(0,1) > 0.5:
                max_length = 5
            else:
                max_length = self.max_length
            max_length = np.random.choice(np.arange(2, max_length), 1)[0]
      
                
            if len(user_seq) > max_length:
                start = np.random.choice(np.arange(0, len(user_seq)-max_length), 1)[0]
                sample_user_seq = user_seq[start:start+max_length]
                target_item = user_seq[start+max_length]
            else:
                sample_user_seq = user_seq[:-1]
                target_item = user_seq[-1]
        else:
            start = -self.max_length-1 if len(user_seq) > self.max_length else 0
            sample_user_seq = user_seq[start:-1]
            target_item = user_seq[-1]
        
        # User History
        valid_context_length = True
        p = 1
        while valid_context_length:
            seq_img = []
            prompt_seq = ""
            seq_id = []
            for i, seq in enumerate(sample_user_seq):
                seq_id.append(int(seq[0]))
                
                # Item Info
                i_prompt = self.prompt_per_item(seq[0])
                prompt_seq += i_prompt
                
                if not self.rec_args.load_features:
                    img_path = os.path.join(self.image_folder, f"{seq[0]}.jpg")
                    i_img = process_image(img_path, self.data_args)
                    seq_img.append(i_img)
                    
            if not self.rec_args.load_features:       
                seq_img = torch.stack(seq_img)
            
            if len(self.tokenizer(prompt_seq).input_ids) < 3850: # Even if the context length size is 4,096, we have to put the instruction
                valid_context_length = False
            else:
                if self.split=='train':
                    if len(user_seq) > max_length:
                        sample_user_seq = user_seq[start+p:start+max_length]
                    else:
                        sample_user_seq = user_seq[p:-1]
                    p += 1
                else:
                    start += 1
                    sample_user_seq = user_seq[start:-1]
        
        prompt_seq = prompt_seq[:-2]
        prompt_seq += "]. "
        input_prompt = task_prompt + prompt_seq
        input_prompt = input_prompt[:-2] + ", where each interacted item's information is contained in curly brackets (i.e.,{}). "
        history_prompt = copy.deepcopy(input_prompt)
        IRE_prompt = input_prompt + conversation_lib.IRE_prompt
        return IRE_prompt, history_prompt, seq_img, seq_id, target_item[0]


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        IRE_prompt, history_prompt, seq_img, seq_id, target_item = self.process_IRE_prompt(i)
        ILA_sources = self.construct_ILA_prompt(target_item, history_prompt)
        
        #if self.print_first:
        #    print(f"Human: {sources[0][0]['value']} / Assistant: {sources[0][1]['value']}")
        #    self.print_first = False
            
        IRE_sources = [[{'from':'human', 'value': IRE_prompt}, {'from': 'gpt', 'value': ""}]]
        
        IRE_input_ids = self.tokenizer(IRE_sources[0][0]['value'], return_tensors='pt', padding='longest', max_length=self.tokenizer.model_max_length, truncation=True).input_ids
        labels = torch.zeros(IRE_input_ids.shape) -100
        IRE_data_dict = {'input_ids': IRE_input_ids,'labels': labels}

        data_dict = dict(IRE_input_ids=IRE_data_dict["input_ids"][0], labels=IRE_data_dict["labels"][0])
        ILA_data_dict = dict(input_ids=ILA_sources["input_ids"][0], labels=ILA_sources["labels"][0])

        data_dict['ILA_input_ids'] = ILA_data_dict['input_ids']
        data_dict['ILA_labels'] = ILA_data_dict['labels']
        data_dict['seq_img'] = seq_img
        data_dict['seq_id'] = seq_id
        data_dict['target_id'] = target_item
        data_dict['user_name'] = self.user_idx2name[i]
        return data_dict
        

    def construct_ILA_prompt(self, target_id, history_prompt):
        random_question = np.random.choice(['title', 'feature', 'brand', 'category'], 1)[0]
        random_number = np.random.choice(np.arange(5), 1)[0]
        question = f"Q_{random_question}_{random_number}"
        question = f"Please conduct an analysis of user preference, taking into account the interacted items. Based on your analysis, {conversation_lib.ILA_prompt_template[question]}"
        answer = self.get_meta_info(target_id, random_question)
        
        if random_question == 'feature' and len(answer) > 1:
            answer = np.random.choice(answer, 1)[0]
            
        answer = conversation_lib.ILA_prompt_template[f"A_{random_question}"] + f"'{answer}'"
        ILA_prompt = history_prompt + question
        ILA_sources = [[{'from':'human', 'value': ILA_prompt}, {'from': 'gpt', 'value': answer}]]
        ILA_sources = preprocess(ILA_sources, self.tokenizer, has_image=False, train=self.split)
        return ILA_sources
        


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments
    train: str
    generation_kwargs: dict
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        IRE_input_ids = [instance['IRE_input_ids'] for instance in instances]
        seq_id = [instance['seq_id'] for instance in instances]
        target_id = [instance['target_id'] for instance in instances]
        seq_img = [instance['seq_img'] for instance in instances]
        ILA_input_ids = [instance['ILA_input_ids'] for instance in instances]
        ILA_labels = [instance['ILA_labels'] for instance in instances]
        user_name = [instance['user_name'] for instance in instances]
        IRE_input_ids = torch.nn.utils.rnn.pad_sequence(
            IRE_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
            
        ILA_input_ids = torch.nn.utils.rnn.pad_sequence(
            ILA_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        ILA_labels = torch.nn.utils.rnn.pad_sequence(ILA_labels, batch_first=True, padding_value=IGNORE_INDEX)         
            
        # IRE: For Recommendation
        IRE_input_ids = IRE_input_ids[:, : self.tokenizer.model_max_length]
        IRE_attention_mask = IRE_input_ids.ne(self.tokenizer.pad_token_id)
        
        # ILA: For alignment
        ILA_input_ids = ILA_input_ids[:, :self.tokenizer.model_max_length]
        ILA_labels = ILA_labels[:,:self.tokenizer.model_max_length]
        ILA_attention_mask = ILA_input_ids.ne(self.tokenizer.pad_token_id)
        
        batch = dict(
            IRE_input_ids=IRE_input_ids,
            IRE_attention_mask=IRE_attention_mask,
            seq_img = seq_img,
            seq_id=seq_id,
            target_id = target_id,
            train=self.train,
            user_name=user_name,
            ILA_input_ids=ILA_input_ids,
            ILA_labels=ILA_labels,
            ILA_attention_mask=ILA_attention_mask,
            generation_kwargs=self.generation_kwargs
        )
        return batch


def load_dataset(args, rec_args, tokenizer, train=True, generation_kwargs=None):
    data_path = args.data_path
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=args, train=train, generation_kwargs=generation_kwargs)
    if train:
        train_dataset = LazyRecDataset(data_path=data_path, tokenizer=tokenizer, rec_args=rec_args, data_args=args, split='train')
        return dict(train_dataset=train_dataset, data_collator=data_collator)
    else:
        valid_dataset = LazyRecDataset(data_path=data_path, tokenizer=tokenizer, rec_args=rec_args, data_args=args, split='test')
        return dict(eval_dataset=valid_dataset, data_collator=data_collator)
  