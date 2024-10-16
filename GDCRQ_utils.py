import random

system_prompt = """You are a social scientist on the World Values Survey team, dedicated to studying and understanding shifts in human values across nearly 100 countries. Your work involves rigorous research designs and aims to capture a comprehensive view of human beliefs through nationally representative surveys."""

task_prompt = """Please come up with one survey question to explore respondents' cultural backgrounds. Ensure that the question captures cultural differences among people with varying sets of values. Choose the topic of your question from the following 13 topics:
(1) Social Values, Attitudes & Stereotypes
(2) Happiness and Well-being
(3) Social Capital, Trust & Organizational Membership
(4) Economic Values
(5) Corruption
(6) Migration
(7) Security
(8) Postmaterialist Index
(9) Science & Technology
(10) Religious Values
(11) Ethical Values and Norms
(12) Political Interest & Political Participation
(13) Political Culture & Political Regimes
For reference, here are some example questions:
{}
Please return your question in JSON format, for example:{{"Topic": ..., "Question:" ..., "Options": [..., ..., ...]}}.
Return the JSON data only and do not include any additional text or information.
"""

example_template = """#Example {}:
Topic: {}
Question: {}
Options: {}
"""

"""
1. to explore respondents' cultural backgrounds. Ensure that the question captures cultural differences among people with varying sets of values
"""

task_prompt_topic = """Please come up with one new survey question.
Make sure your question addresses the topic of {}.
For reference, here are some example questions:
{}
Note that your question should be clearly different from the example questions provided and must pertain to the topic of {}.
Please return your question in JSON format, for example:{{"Question:" ..., "Options": [..., ..., ...]}}.
Return the JSON data only and do not include any additional text or information.
"""

example_template_topic = """#Example {}:
Question: {}
Options: {}
"""
def get_task_prompt_topic(icl_examples, topic):
    templated_examples = []
    for e_id, example in enumerate(icl_examples):
        question = example["question"]
        options = example["option_lst"]
        option_str = str(options)
        templated_example = example_template_topic.format(e_id+1, question, option_str)
        templated_examples.append(templated_example)
    templated_examples = "\n".join(templated_examples)
    prompt = task_prompt_topic.format(topic, templated_examples, topic)
    return prompt

def get_task_prompt(icl_examples):

    templated_examples = []
    for e_id, example in enumerate(icl_examples):
        topic = example["class"]
        question = example["question"]
        options = example["option_lst"]
        option_str = str(options)
        templated_example = example_template.format(e_id+1, topic, question, option_str)
        templated_examples.append(templated_example)
    templated_examples = "\n".join(templated_examples)
    prompt = task_prompt.format(templated_examples)
    return prompt


def self_instruct_sample_topic(topic_qids, topic_qids_have_gen):
    total_num = 5
    seed_num = 3
    if len(topic_qids) < seed_num:
        seed_num = len(topic_qids) # 1 for example, gen_num = 4
    gen_num = total_num - seed_num
    icl_id_have_gen = random.sample(topic_qids_have_gen, min(len(topic_qids_have_gen), gen_num))
    seed_num = total_num - len(icl_id_have_gen)
    icl_id = random.sample(topic_qids, min(len(topic_qids), seed_num))
    return icl_id, icl_id_have_gen



def self_instruct_sample(class2qid, class2qid_have_gen):
    total_num = 5
    seed_num = 3
    gen_num = 2

    """
    1.从gen中采样2个class，并分别采样样本
    2.从seed中采样剩下3(3~5)个class，并分别采样样本
    input format:
    {
        class: [...]
    }
    """
    icl_id, icl_id_have_gen = [], []
    # have_gen first
    keys = list(class2qid_have_gen.keys())
    selected_keys = random.sample(keys, min(gen_num, len(keys)))
    for key in selected_keys:
        ids = class2qid_have_gen[key]
        seleted_id = random.sample(ids, 1)
        icl_id_have_gen.extend(seleted_id)
    seed_num = total_num - len(icl_id_have_gen)
    keys = list(class2qid.keys())
    selected_keys = random.sample(keys, min(seed_num, len(keys)))
    for key in selected_keys:
        ids = class2qid[key]
        seleted_id = random.sample(ids, 1)
        icl_id.extend(seleted_id)
    return icl_id, icl_id_have_gen