import json
seed_setting = "all"
path = "./data/Meta-Llama-3-8B-Instruct/generated_questions_topic_{idx}_from_{seed_setting}_seed.json"
save_path = f"./data/Meta-Llama-3-8B-Instruct/generated_questions_from_{seed_setting}_seed.json"
all_data = []
q_id = 0
for idx in range(13):
    this_path = path.format(idx=idx, seed_setting=seed_setting)
    with open(this_path, 'r') as file:
        data = json.load(file)
        for one_data in data:
            one_data["Q_id"] = f"Q{q_id}"
            q_id += 1
        all_data.extend(data)

with open(save_path, 'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)

print(save_path)