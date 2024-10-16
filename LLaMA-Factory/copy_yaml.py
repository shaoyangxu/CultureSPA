import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from collections import OrderedDict
path = "/data/syxu/culture_steering/LLaMA-Factory/llama3_lora_sft_all.yaml"
# 读取 YAML 文件

with open(path, 'r') as file:
    data = yaml.safe_load(file)

for country, culture in country2culture_dict.items():
    new_data = dict()
    for k, v in data.items():
        if k == "dataset":
            v = f"culture_{country}"
        elif k == "output_dir":
            v = f"saves/llama3_lora_sft_{country}"
            if country == "USA":
                v += "_2"
        new_data[k] = v
    new_data["per_device_train_batch_size"] = 1
    new_data["gradient_accumulation_steps"] = 8
    # print(new_data)
    # exit()
    save_path = f"/data/syxu/culture_steering/LLaMA-Factory/llama3_lora_sft_{country}.yaml"
    with open(save_path, 'w') as file:
        yaml.dump(new_data, file, default_flow_style=False)
    print(f"CUDA_VISIBLE_DEVICES=0 llamafactory-cli train llama3_lora_sft_{country}.yaml")