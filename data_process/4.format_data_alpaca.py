import json
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from result_analysis_run_3 import parse_answer
en_culture_system_prompt = system_prompt_dict["en_culture_aware"]
seed_setting = "all"

question_path = f"./data/Meta-Llama-3-8B-Instruct/generated_from_{seed_setting}_seed_result_culture_aware_inconsistent.json"

save_path = f"./LLaMA-Factory/data/CultureSPA_culture_aware.json"

with open(question_path, 'r') as file:
    questions = json.load(file)
    lang = "en"

train_datas = []

for question in questions:

    country = question["country"]

    culture = country2culture_dict[country]
    a_an = a_an_dict[country]
    system_prompt = en_culture_system_prompt.format(a_an, culture)

    q_id = question["Q_id"]
    question_str = question["question"]
    option_lst = question["option_lst"]
    answer = parse_answer(question["answer"])
    assert answer != "None"
    question_str = question_template_generated_questions(q_id, question_str, option_lst, lang)
    train_data = {
        "instruction": question_str,
        "input": "",
        "output": answer,
        "system": system_prompt
    }
    train_datas.append(train_data)

print(len(train_datas))
with open(save_path, 'w', encoding='utf-8') as file:
    json.dump(train_datas, file, ensure_ascii=False, indent=4)

print(save_path)