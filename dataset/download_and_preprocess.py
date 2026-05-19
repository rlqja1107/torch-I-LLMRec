# https://github.com/weitianxin/UniMP/tree/main/data

import os
import sys
sys.path.append(".")
import json
import pickle
import random
import requests
import argparse
import numpy as np
from tqdm import tqdm
from urllib import request
from data_name import data_dict
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="Art")
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()

user_core= args.user_core
print(args.data_name, ":", user_core)
save_name = args.data_name

item_core=5
process_dir = f"./dataset/Amazon_18/{save_name}"
os.makedirs(process_dir, exist_ok=True)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
# meta data extraction
def extract_meta(data_name, meta_data):
    data_name = "_".join(data_name.split(" "))
    print("Extract Meta", data_name)
    meta_path = f"{process_dir}/meta_{data_dict[data_name]}.json"
    if not os.path.exists(meta_path):
        os.system(f"wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_{data_dict[data_name]}.json.gz")
        # https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Arts_Crafts_and_Sewing.json.gz
        os.system(f"gzip -d meta_{data_dict[data_name]}.json.gz")
        os.system(f"mv meta_{data_dict[data_name]}.json {process_dir}")
    num1,num2,num3,num4,num5 = 0,0,0,0,0
    num_noimage=0
    with open(meta_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            dict_line = eval(line)
            attr_dict = {}
            if ("imageURL" in dict_line and len(dict_line['imageURL']) > 0) or ("imageURLHighRes" in dict_line and len(dict_line['imageURLHighRes']) > 0):
                if "imageURLHighRes" in dict_line and len(dict_line['imageURLHighRes']) > 0:
                    attr_dict["imURL"] = dict_line["imageURLHighRes"][0]
                else:
                    attr_dict["imURL"] = dict_line["imageURL"][0]
                
                if "category" in dict_line and len(dict_line['category']) > 0:
                    category = dict_line['category'][0]
                    attr_dict['category'] = category
                else:
                    attr_dict['category'] = ""
                    num1+=1
                if "brand" in dict_line:
                    brand = dict_line['brand']
                    attr_dict['brand'] = brand
                else:
                    attr_dict['brand'] = ""
                    num2+=1
                if "title" in dict_line:
                    title = dict_line['title']
                    attr_dict['title'] = title
                else:
                    attr_dict['title'] = ""
                    num3+=1
                if "description" in dict_line and len(dict_line['description']) > 0:
                    des = dict_line['description'][0]
                    attr_dict['description'] = des
                else:
                    attr_dict['description'] = ""
                    num4+=1
                if "price" in dict_line and dict_line['price'] != '':
                    price = dict_line['price']
                    attr_dict['price'] = price
                else:
                    attr_dict['price'] = ""
                    num5+=1
                if "feature" in dict_line:
                    feature = dict_line['feature']
                    attr_dict['feature'] = feature
                else:
                    attr_dict['feature'] = ""
                    num5+=1
                asin = dict_line["asin"]
                meta_data[asin] = attr_dict
    print(num_noimage,num1, num2, num3, num4,num5)
    return meta_data

meta_data={}
meta_data=extract_meta(save_name, meta_data=meta_data)

def extract_interaction(data_name, sequences, asin_set):
    data_name = "_".join(data_name.split(" "))
    print("Extract Interactions", data_name)
    inter_path = f"{process_dir}/{data_dict[data_name]}_5.json"
    if not os.path.exists(inter_path):
        os.system(f"wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/{data_dict[data_name]}_5.json.gz")
        # https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz
        os.system(f"gzip -d {data_dict[data_name]}_5.json.gz")
        os.system(f"mv {data_dict[data_name]}_5.json {process_dir}")
    user_set, item_set, inter_num = set(), set(), 0
    exp_path = 'raw_data/reviews_{}.pickle'.format(data_name)
    if os.path.exists(exp_path):
        raw_explanations = load_pickle(exp_path)
        use_exp=True
    else:
        use_exp=False
    with open(inter_path,"r") as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            line = line.replace('"verified": true', '"verified": True')
            line = line.replace('"verified": false', '"verified": False')
            dict_line = eval(line.strip("\n"))
            
            # add explanation only for partial datasets
            if use_exp:
                raw_explanation = raw_explanations[i]
                assert dict_line['reviewerID'] == raw_explanation['user']
                assert dict_line['asin'] == raw_explanation['item']
                if 'sentence' in raw_explanation:
                    list_len = len(raw_explanation['sentence'])
                    selected_idx = random.randint(0, list_len-1)
                    explanation = raw_explanation['sentence'][selected_idx][2]
                else:
                    explanation = ""
            else:
                explanation = ""
            # add end
            user = dict_line['reviewerID']
            asin = dict_line['asin']
            time = dict_line['unixReviewTime']
            if hasattr(dict_line, "reviewText"):
                review = dict_line['reviewText']
            else:
                review = ""
            rate = dict_line['overall']

            if hasattr(dict_line, "summary"):
                summary = dict_line['summary']
            else:
                summary = ""

            if asin in meta_data:
                sequences[user+'_'+data_name].append([time, asin, explanation, rate, summary, review])
                user_set.add(user)
                item_set.add(asin)
                inter_num += 1
                asin_set.add(asin)                
        print(f'Dataset: {data_name}, User: {len(user_set)}, Items: {len(item_set)}, Interaction numbers: {inter_num} asin_set: {len(asin_set)}')

    return sequences, asin_set


training_sequences = defaultdict(list)
asin_set = set()
training_sequences, asin_set = extract_interaction(save_name, training_sequences, asin_set)
import copy
def post_process(sequences):
    length = 0
    for user, sequence in tqdm(sequences.items()):
        sequences[user] = [ele[1:] for ele in sorted(sequence)]
        length += len(sequences[user])

    print(f'Averaged length: {length/len(sequences)}')

    return sequences

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


training_sequences = post_process(training_sequences)
training_sequences = filter_Kcore(training_sequences, user_core=user_core, item_core=item_core)

asin_set = set()
for user, items in training_sequences.items():
    for item in items:
        asin_set.add(item[0])
print("filter user size:", len(training_sequences), "filter item size:", len(asin_set))
meta_data = {asin: meta_data[asin] for asin in asin_set}
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
num = int(len(keys)*0.8)
num1 = int(len(keys)*0.9)
#train_keys, eval_keys, test_keys = keys[:num], keys[num: num1], keys[num1:]
train_data = {key: new_data[key][:-2] for key in keys}
eval_data = {key: new_data[key][:-1] for key in keys}
test_data = {key: new_data[key][:] for key in keys}


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

    

def down_save(url, image_name):
    r = requests.get(url, stream=True)
    with open(image_name, 'wb') as f:
        f.write(r.content)

with open(f'{process_dir}/meta_info.json', 'r') as f:
    meta_data = json.load(f)
if not os.path.exists(f"{process_dir}/{save_name}"):
    os.mkdir(f"{process_dir}/{save_name}")
    

# Remove original file
os.system(f"rm {process_dir}/meta_{data_dict[save_name]}.json")
os.system(f"rm {process_dir}/{data_dict[save_name]}_5.json")