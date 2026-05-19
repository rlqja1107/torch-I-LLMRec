import os
import json
import requests
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="Art")
parser.add_argument("--user_core", type=int, default=5)
args = parser.parse_args()
item_core=5
save_name = args.data_name

user_core= args.user_core
print(save_name, ":", user_core)
process_dir = f"./dataset/Amazon_18/{save_name}"
os.makedirs(process_dir, exist_ok=True)

def down_save(url, image_name):
    r = requests.get(url, stream=True)
    with open(image_name, 'wb') as f:
        f.write(r.content)

with open(f'{process_dir}/meta_info.json', 'r') as f:
    meta_data = json.load(f)
if not os.path.exists(f"{process_dir}/{save_name}"):
    os.mkdir(f"{process_dir}/{save_name}")
for key, values in tqdm(meta_data.items()):
    imUrl = values["imURL"]
    out_filepath=f"{process_dir}/{save_name}/{key}.jpg"
    try:
        down_save(imUrl, out_filepath)
    except:
        print("bad")