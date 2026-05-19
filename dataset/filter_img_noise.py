import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="Art")
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
data_name = args.data_name
user_core= args.user_core

process_dir = f"./dataset/Amazon_18/{data_name}"
previous_img_path = data_name
save_name="image"

print("Filtering Noise Image")

num_items = len(json.load(open(f"{process_dir}/meta_info.json", 'r')))
existing_imgs = [int(_.split(".")[0]) for _ in os.listdir(f"{process_dir}/{previous_img_path}")]
missing_imgs = []
good_imgs = []
for img_id in existing_imgs:
    try:
        img_ = Image.open(os.path.join(f"{process_dir}/{previous_img_path}", f"{img_id}.jpg")).convert("RGB")
        good_imgs.append(img_id)
    except:
        missing_imgs.append(img_id)
print(missing_imgs)
print(f"{len(missing_imgs)} are broken")

with open(f"{process_dir}/test_users.json", 'r') as f:
    training_sequences = json.load(f)

with open(f"{process_dir}/meta_info.json", 'r') as f:
    meta_data = json.load(f)
error_description_list = []
for k, v in meta_data.items():
    if len(v['description']) == 0:
        error_description_list.append(int(k))

good_imgs = set(good_imgs) - set(error_description_list)
good_imgs = list(good_imgs)
good_imgs_dict = {img_id:0 for img_id in good_imgs}
for user, full_seqs in training_sequences.items():
    pop_indexs = []
    new_full_seqs = [full_seq for i, full_seq in enumerate(full_seqs) if full_seq[0] in good_imgs_dict]
    training_sequences[user] = new_full_seqs


# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item[0]] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # Kcore

# K-core
def filter_Kcore(user_items, user_core, item_core): # user items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # user
                user_items.pop(user)
            else:
                for full_item in user_items[user]:
                    item = full_item[0]
                    if item_count[item] < item_core:
                        item_user = [full_item[0]==item for full_item in user_items[user]]
                        index = np.where(item_user)[0][0]
                        user_items[user].pop(index)
                        # user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items
training_sequences = filter_Kcore(training_sequences, user_core=user_core, item_core=5)

remove_list = []
for k, v in training_sequences.items():
    if len(v) < 3:
        remove_list.append(k)
for r in remove_list:
    del training_sequences[r]

asin_set = set()
for user, items in training_sequences.items():
    for item in items:
        asin_set.add(item[0])
print("filter user size:", len(training_sequences), "filter item size:", len(asin_set))
meta_data = {asin: meta_data[str(asin)] for asin in asin_set}
# reorder
asin2id={}
id=0
for user, values in training_sequences.items():
    asins = [value[0] for value in values]
    for asin in asins:
        asin2id.setdefault(asin, id)
        if asin2id[asin]==id:
            id+=1
keys = list(asin2id.keys())

values = list(asin2id.values())
import random, copy
old_values = copy.deepcopy(values)
random.seed(42)
random.shuffle(values)
for key, value in zip(keys, values):
    asin2id[key] = value
# resave images
import os
new_data, new_meta_data = copy.deepcopy(training_sequences), {}
for user, values in training_sequences.items():
    for i, value in enumerate(values):
        new_data[user][i][0] = asin2id[value[0]]
for asin, attr in meta_data.items():
    id = asin2id[asin]
    new_meta_data[id] = attr


keys = list(new_data.keys())
random.seed(42)
random.shuffle(keys)
train_data = {key: new_data[key][:-2] for key, value in new_data.items()}
eval_data = {key: new_data[key][:-1] for key, value in new_data.items()}
test_data = {key: new_data[key] for key, value in new_data.items()}

if not os.path.exists(f"{process_dir}"):
    os.mkdir(f"{process_dir}")
with open(f'{process_dir}/users.json', 'w') as f:
    json.dump(new_data, f)
with open(f'{process_dir}/train_users.json', 'w') as f:
    json.dump(train_data, f)
with open(f'{process_dir}/eval_users.json', 'w') as f:
    json.dump(eval_data, f)
with open(f'{process_dir}/test_users.json', 'w') as f:
    json.dump(test_data, f)
with open(f'{process_dir}/meta_info.json', 'w') as f:
    json.dump(new_meta_data, f)

    
if not os.path.exists(f"{process_dir}/{save_name}"):
    os.mkdir(f"{process_dir}/{save_name}")
    
print("Download Images")
for img_id in tqdm(asin_set):
    new_img_id = asin2id[img_id]
    img = Image.open(f"{process_dir}/{previous_img_path}/{img_id}.jpg").convert("RGB")
    img.save(f"{process_dir}/{save_name}/{new_img_id}.jpg")
