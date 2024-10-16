from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import random
import numpy as np
from utils import *
import os
import fire

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(device_id, model_path="", lora_name="", culture_specific=False, run_id=0):
    set_seed(seed=42-run_id)

    lang = "en"
    question_path = "./data/wvs_questions.json"

    device = torch.device(f"cuda:{device_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    with open(question_path, 'r') as file:
        questions = json.load(file)

    if "wvs_questions" in question_path:
        questions = wvs_question_filter(questions)

    if not culture_specific:
        if lora_name:
            model_name = lora_name
            lora_path = f"./LLaMA-Factory/saves/{lora_name}"
            model = load_lora(model, lora_path=lora_path)
        else:
            model_name = model_path.split("/")[-1]
        save_dir = f"./result/{model_name}"
        
    questions_dict = {q["Q_id"]: {'q': q["question"], 'o': q["option_lst"]} for q in questions}

    system_prompt = system_prompt_dict["en"]
    for country, culture in country2culture_dict.items():
        a_an = a_an_dict[country]
        if culture_specific:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            ).to(device)
            model_name = f"llama3_lora_sft_{country}"
            save_dir = f"./result/{model_name}"
            lora_path = f"./LLaMA-Factory/saves/{model_name}"
            lora_model = load_lora(model, lora_path=lora_path)
        os.makedirs(save_dir, exist_ok=True)
        if "wvs_questions" in question_path:
            save_path = f"{save_dir}/wvs_result_self_alignment_{run_id}.json"
        result = []
        for question in tqdm(questions):
            q_id = question["Q_id"]
            question_str = question["question"]
            option_lst = question["option_lst"]
            question_str = question_template_self_alignment(q_id, question_str, option_lst, lang, country, questions_dict)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_str},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict = True,
                return_tensors = 'pt'
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            if culture_specific:
                outputs = lora_model.generate(
                    input_ids = input_ids['input_ids'],
                    attention_mask = input_ids['attention_mask'],
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    input_ids = input_ids['input_ids'],
                    attention_mask = input_ids['attention_mask'],
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = outputs[0][input_ids['input_ids'].shape[-1]:]
            response_str = tokenizer.decode(response, skip_special_tokens=True)
            question["answer"] = response_str
            result.append(question)
        
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(questions, file, ensure_ascii=False, indent=4)

        print(save_path)

if __name__ == "__main__":
    fire.Fire(main)