import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from collections import OrderedDict
path = "/data/syxu/culture_steering/LLaMA-Factory/llama3_lora_sft_all.yaml"
# 读取 YAML 文件

country2culture_dict = {
    "all": "all",
    "USA": "American",
    "CAN": "Canadian",
    "BOL": "Bolivian",
    "BRA": "Brazilian",
    "GBR": "British",
    "NLD": "Dutch",
    "DEU": "German",
    "UKR": "Ukrainian",
    "CHN": "Chinese",
    "RUS": "Russian",
    "IND": "Indian",
    "THA": "Thai",
    "KEN": "Kenyan",
    "NGA": "Nigerian",
    "ETH": "Ethiopian",
    "ZWE": "Zimbabwean",
    "AUS": "Australian",
    "NZL": "New Zealand",
}

with open(path, 'r') as file:
    data = yaml.safe_load(file)

ratio_lst = [float(f'0.{i}') for i in range(1, 10)]

for ratio in ratio_lst:
    dir_path = f"/data/syxu/culture_steering/LLaMA-Factory/{ratio}"
    os.makedirs(dir_path, exist_ok=True)

    for country, culture in country2culture_dict.items():
        new_data = dict()
        for k, v in data.items():
            if k == "dataset":
                v = f"culture_{country}{ratio}"
            elif k == "output_dir":
                v = f"/data/syxu/culture_steering/LLaMA-Factory/saves/{ratio}/llama3_lora_sft_{country}"
            new_data[k] = v
        new_data["per_device_train_batch_size"] = 4
        new_data["gradient_accumulation_steps"] = 32
        save_path = f"{dir_path}/llama3_lora_sft_{country}.yaml"
        with open(save_path, 'w') as file:
            yaml.dump(new_data, file, default_flow_style=False)
        print(f"CUDA_VISIBLE_DEVICES=0 llamafactory-cli train llama3_lora_sft_{country}.yaml")