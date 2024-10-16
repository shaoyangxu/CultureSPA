import json
import os
import re
import numpy as np
from utils import country2culture_dict, zero_questions, wvs_question_filter
import random
random.seed(42)

def process_gold(gold_dict):
    ret_gold_dict = {}
    for q_id, info in gold_dict.items():
        try:
            ret_gold_dict[q_id] = list(info.keys())[0]
        except:
            continue
    return ret_gold_dict

def parse_answer(answer):
    match = re.search(r'\d+', answer)
    if match:
        number = match.group()
        if 0 <= int(number) <= 10:
            return number
    return "None"

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def alignment_score(normed_distance):
    score = 1 - normed_distance
    return score

def compare_res_gold(res_lst, gold_dict, question_list, lang, gold_lang):
    all_cot = len(res_lst)
    valid_cot = 0

    no_valid_lst = []
    gold_points = []
    res_points = []
    q_id_lst = []
    for res in res_lst:
        q_id = res["Q_id"]
        answer = parse_answer(res["answer"])
        if answer == "None":
            no_valid_lst.append(res)
            continue
        if q_id not in gold_dict: # 'Q215'-en
            continue
        gold_answer = gold_dict[q_id]
        q_id_lst.append(q_id)
        valid_cot += 1
        res_points.append(int(answer))
        gold_points.append(int(gold_answer))

    # print(f"res({lang})-gold({gold_lang}) | valid({valid_cot})-all({all_cot})")
    distance = euclidean_distance(res_points, gold_points) / compute_max_distance(question_list, q_id_lst)
    score = alignment_score(distance)
    # print(score)
    return score

def compare_random_gold(res_lst, gold_dict, question_list, lang, gold_lang):
    all_cot = len(res_lst)
    valid_cot = 0

    no_valid_lst = []
    gold_points = []
    res_points = []
    q_id_lst = []
    for res in res_lst:
        q_id = res["Q_id"]
        answer = res["answer"]
        if q_id not in gold_dict: # 'Q215'-en
            continue
        gold_answer = gold_dict[q_id]
        q_id_lst.append(q_id)
        valid_cot += 1
        res_points.append(int(answer))
        gold_points.append(int(gold_answer))

    print(f"random({lang})-gold({gold_lang}) | valid({valid_cot})-all({all_cot})")
    distance = euclidean_distance(res_points, gold_points) / compute_max_distance(question_list, q_id_lst)
    score = alignment_score(distance)
    print(score)
    return score

def compare_res_res(res_lst, res_lst2, question_list):
    all_cot = len(res_lst)
    valid_cot = 0
    no_valid_lst = []
    res_points = []
    res_points2 = []
    q_id_lst = []
    for idx, res in enumerate(res_lst):
        q_id = res["Q_id"]
        answer = parse_answer(res["answer"])
        res2 = res_lst2[idx]
        answer2 = parse_answer(res2["answer"])
        if answer == "None" or answer2 == "None":
            no_valid_lst.append([res, res2])
            continue
        q_id_lst.append(res["Q_id"])
        valid_cot += 1
        res_points.append(int(answer))
        res_points2.append(int(answer2))
    print(f"res-res | valid({valid_cot})-all({all_cot})")
    distance = euclidean_distance(res_points, res_points2) / compute_max_distance(question_list, q_id_lst)
    score = alignment_score(distance)
    print(score)

def compute_max_distance(question_list, q_id_lst):
    id2info = {q['Q_id']: {'q': q['question'], 'o': q['option_lst']} for q in question_list}
    point1 = [1 for q_id in q_id_lst]
    point2 = [len(id2info[q_id]['o']) for q_id in q_id_lst]
    max_distance = euclidean_distance(point1, point2)
    return max_distance

def generate_random_result(question_list, q_id_lst):
    id2info = {q['Q_id']: {'q': q['question'], 'o': q['option_lst']} for q in question_list}
    point1 = [1 for q_id in q_id_lst]
    point2 = [len(id2info[q_id]['o']) for q_id in q_id_lst]
    max_distance = euclidean_distance(point1, point2)
    return max_distance

if __name__ == "__main__":

    lang = "en"
    gold_lang = "en"

    gold_path = "./data/proportions_group_by_country.json"
    question_path = "./data/wvs_questions.json"
    with open(question_path, 'r') as file:
        question_list = json.load(file)
    question_list = wvs_question_filter(question_list)
    with open(gold_path, 'r') as file:
        gold_dict = json.load(file)

    all_score_dict = {}
    run_num = 1
    for repeat_id in range(run_num):
        is_random = False

        if is_random:
            res_lst = []
            for question in question_list:
                q_id = question["Q_id"]
                option_lst = question["option_lst"]
                option_num = len(option_lst)
                if q_id in zero_questions:
                    option_id_lst = [o_id for o_id in range(0, option_num)]
                else:
                    option_id_lst = [o_id for o_id in range(1, option_num+1)]
                random_choice = random.choice(option_id_lst)
                res_lst.append({"Q_id": q_id, "answer": random_choice})

        score_lst = []
        
        # culture_unaware
        res_path = f"result/Meta-Llama-3-8B-Instruct/wvs_result_culture_unaware_{repeat_id}.json"
        for country in country2culture_dict:
            # culture aware prompting
            res_path = f"./result/Meta-Llama-3-8B-Instruct/wvs_result_culture_aware_{country}_{repeat_id}.json"
            # cross culture thinking
            res_path = f"./result/Meta-Llama-3-8B-Instruct/wvs_result_cross_culture_{country}_{repeat_id}.json"
            # self alignment
            res_path = f"./result/Meta-Llama-3-8B-Instruct/wvs_result_self_alignment_{country}_{repeat_id}.json"
            # culture aware prompting + self alignment
            res_path = f"./result/Meta-Llama-3-8B-Instruct/wvs_result_culture_aware_self_alignment_{country}_{repeat_id}.json"
            # cross culture thinking + self alignment
            res_path = f"./result/Meta-Llama-3-8B-Instruct/wvs_result_cross_culture_self_alignment_{country}_{repeat_id}.json"
            if not is_random:
                with open(res_path, 'r') as file:
                    res_lst = json.load(file)


            lang_gold_dict = gold_dict[country]
            lang_gold_dict = process_gold(lang_gold_dict)

            if not is_random:
                score = compare_res_gold(res_lst, lang_gold_dict, question_list, lang, country)
                score_lst.append(score)
            else:
                score = compare_random_gold(res_lst, lang_gold_dict, question_list, lang, country)
                score_lst.append(score)
        all_score_dict[repeat_id] = score_lst
        if repeat_id == 0:
            print(dict(zip(list(country2culture_dict.keys()), score_lst)))
        # print(score_lst)
    avg_score_lst = []
    for idx in range(len(all_score_dict[0])):
        avg_lst = []
        for repeat_id, score_lst in all_score_dict.items():
            avg_lst.append(score_lst[idx])
        avg = sum(avg_lst) / len(avg_lst)
        avg_score_lst.append(avg)
    for idx, country in enumerate(list(country2culture_dict.keys())):
        print(country)
        print(round(avg_score_lst[idx] * 100, 2))
    avg = sum(avg_score_lst) / len(avg_score_lst)
    print("AVG")
    print(round(avg * 100, 2))