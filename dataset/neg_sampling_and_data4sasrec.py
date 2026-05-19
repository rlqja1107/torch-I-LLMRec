# 1. Negative Sampling for evaluation
# 2. Data preprocessing for SASRec training

import os
import json
import argparse
import itertools
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="Art")
args = parser.parse_args()
data_name = args.data_name
print("Negative Sampling for evaluation & Data preprocessing for SASRec")

seq_id = json.load(open(f"./dataset/Amazon_18/{data_name}/users.json", 'r'))
i_meta = json.load(open(f"./dataset/Amazon_18/{data_name}/meta_info.json", 'r'))
max_item = len(i_meta)

remove_idx = []
for k, v in seq_id.items():
    if len(v) < 3: remove_idx.append(k)
    
for r in remove_idx:
    del seq_id[r]

item_set = set()
for k, v in seq_id.items():
    for v1 in v:
        item_set.add(v1[0])
    
# Index starts with 0: Ours
# Index starts with 1: SASRec
sasrec_seq_id_dict = {}
for user_idx, (user, info) in enumerate(seq_id.items()):
    user_seq = [k[0]+1 for k in info]
    sasrec_seq_id_dict[user] = user_seq

# Negative Item Sampling for evaluation
ours_seq_id_dict = {}
for user_idx, (user, info) in enumerate(seq_id.items()):
    user_seq = [k[0] for k in info]
    ours_seq_id_dict[user] = user_seq


neg_item = {}
for k, v in ours_seq_id_dict.items():
    item_idx = []
    for _ in range(100):
        t = np.random.randint(0, max_item)
        while t in v: t = np.random.randint(0, max_item)
        item_idx.append(t)
    neg_item[k] = item_idx
    
# Saving Negative Item
with open(f"./dataset/Amazon_18/{data_name}/neg_item_set.txt", 'w') as f: # Index: Start with 0
    for k, v in neg_item.items():
        f.write(f"{k} {' '.join(map(str, v))}")
        f.write("\n")

# Saving prepreocessed data for SASRec
with open(f"./dataset/Amazon_18/{data_name}/data4sasrec.txt", 'w') as f:
    for k, v in sasrec_seq_id_dict.items():
        for i in v:
            f.write(f"{k} {i}\n")