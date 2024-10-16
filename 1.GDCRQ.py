from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import random
import numpy as np
from utils import wvs_question_filter, get_class, reformat_questions
from GDCRQ_utils import system_prompt, self_instruct_sample_topic, get_task_prompt_topic
from tqdm import tqdm
import os
import fire

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(device_id, model_path="", num_per_topic=1000, seed_setting = "all"):

    set_seed()

    question_path = "./data/wvs_questions.json"

    with open(question_path, 'r') as file:
        questions = json.load(file)

    questions = wvs_question_filter(questions)
    questions = get_class(questions)
    qid2questions, class2qid = reformat_questions(questions)

    device = torch.device(f"cuda:{device_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    topic_num = 13
    max_iterations = topic_num * num_per_topic
    topic_list = list(class2qid.keys())

    if seed_setting != "all":
        new_class2qid = {}
        for topic, qids in class2qid.items():
            if seed_setting == "one":
                selected_qids = random.sample(qids, 1)
            new_class2qid[topic] = selected_qids
        class2qid = new_class2qid

    model_name = model_path.split("/")[-1]
    save_dir = f"./data/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    with tqdm(total=max_iterations, desc="Processing Topics") as pbar:
        for topic_idx, topic in enumerate(topic_list):
            pbar.set_description(f"Processing Topic {topic_idx + 1}/{len(topic_list)}")
            save_path = f"{save_dir}/generated_questions_topic_{topic_idx}_from_{seed_setting}_seed.json"
            save_interval = 10
            if not os.path.exists(save_path):
                with open(save_path, 'w') as file:
                    json.dump([], file)
            with open(save_path, 'r') as file:
                have_gen = json.load(file)
            
            for _ in have_gen: pbar.update(1)

            in_topic_id = 0
            if len(have_gen) != 0:
                in_topic_id = int(have_gen[-1]['Q_id'][1:]) + 1
            qid2questions_have_gen, class2qid_have_gen = reformat_questions(have_gen)

            while len(have_gen) < num_per_topic:

                topic_qids = class2qid[topic]
                topic_qids_have_gen = [] if len(class2qid_have_gen) == 0 else class2qid_have_gen[topic]
                icl_id, icl_id_have_gen = self_instruct_sample_topic(topic_qids, topic_qids_have_gen)
                icl_examples = [qid2questions[id_] for id_ in icl_id]
                icl_examples_have_gen = [qid2questions_have_gen[id_] for id_ in icl_id_have_gen]
                all_icl_examples = icl_examples + icl_examples_have_gen
                random.shuffle(all_icl_examples)
                prompt = get_task_prompt_topic(all_icl_examples, topic)
                messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
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

                outputs = model.generate(
                    input_ids = input_ids['input_ids'],
                    attention_mask = input_ids['attention_mask'],
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
                resp = outputs[0][input_ids['input_ids'].shape[-1]:]
                resp_str = tokenizer.decode(resp, skip_special_tokens=True)
                try:
                    resp_json = json.loads(resp_str)
                    q, o = None, None
                    q = resp_json["Question"] if "Question" in resp_json else resp_json["question"]
                    o = resp_json["Options"] if "Options" in resp_json else resp_json["options"]
                except:
                    continue
                qid2questions_have_gen[f"Q{in_topic_id}"] = {
                    "Q_id": f"Q{in_topic_id}",
                    "question": q,
                    "option_lst": o,
                    "class": topic,
                    "prompt": prompt
                }
                if topic not in class2qid_have_gen:
                    class2qid_have_gen[topic] = []
                class2qid_have_gen[topic].append(f"Q{in_topic_id}")

                have_gen.append(qid2questions_have_gen[f"Q{in_topic_id}"])
                in_topic_id += 1
                if len(have_gen) % save_interval == 0:
                    with open(save_path, 'w', encoding='utf-8') as file:
                        json.dump(have_gen, file, ensure_ascii=False, indent=4)
                pbar.update(1)

            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(have_gen, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    fire.Fire(main)