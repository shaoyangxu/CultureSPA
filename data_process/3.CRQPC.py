import json
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

country_lst = list(country2culture_dict.keys())

diff_lst = []
all_diff = 0
seed_setting = "all"

save_path = f"./data/Meta-Llama-3-8B-Instruct/generated_from_{seed_setting}_seed_result_culture_aware_inconsistent.json"
for country in country_lst:

    culture_aware = f"./data/Meta-Llama-3-8B-Instruct/generated_from_{seed_setting}_seed_result_culture_aware_{country}.json"

    culture_unaware = f"./data/Meta-Llama-3-8B-Instruct/generated_from_{seed_setting}_seed_result_culture_unaware.json"

    with open(culture_aware, 'r') as file:
        culture_aware_datas = json.load(file)

    with open(culture_unaware, 'r') as file:
        culture_unaware_datas = json.load(file)

    def parse_answer(answer):
        match = re.search(r'\d+', answer)
        if match:
            number = match.group()
            if 0 <= int(number) <= 15:
                return number
        return "None"

    valid, same, diff = 0, 0, 0
    topic_dis = {}
    for idx, culture_aware_data in enumerate(culture_aware_datas):
        culture_unaware_data = culture_unaware_datas[idx]

        culture_aware_ans = parse_answer(culture_aware_data["answer"])
        culture_unaware_ans = parse_answer(culture_unaware_data["answer"])
        if culture_aware_ans != "None" and culture_unaware_ans != "None":
            valid += 1
            if culture_aware_ans == culture_unaware_ans:
                same += 1
            else:
                culture_aware_data["country"] = country
                diff += 1
                diff_lst.append(culture_aware_data)
                class_ = culture_aware_data["class"]
                if class_ not in topic_dis:
                    topic_dis[class_] = 0
                topic_dis[class_] += 1
    print(f"country:{country} | valid:{valid} | same:{same} | diff:{diff} | diff_rate: {diff/valid*100}")
    all_diff += diff

with open(save_path, 'w', encoding='utf-8') as file:
    json.dump(diff_lst, file, ensure_ascii=False, indent=4)

print(all_diff)
print(save_path)