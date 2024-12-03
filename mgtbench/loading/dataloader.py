import random
import datasets
from datasets import load_dataset
import tqdm
import pandas as pd
import re
import copy
import os
import json
from concurrent.futures import ThreadPoolExecutor

# you can add more datasets here and write your own dataset parsing function
DATASETS = ['TruthfulQA', 'SQuAD1', 'NarrativeQA', "Essay", "Reuters", "WP"]

MODELS = ['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini']

CATEGORIES = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 'Economy', 'Math', 'Statistics', 'Chemistry']

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

TOPIC_MAPPING = {
    'Physics': 'STEM',
    'Math': 'STEM',
    'Chemistry': 'STEM',
    'Biology': 'STEM',
    'Electrical_engineering': 'STEM',
    'Computer_science': 'STEM',
    'Statistics': 'STEM',
    'Medicine': 'STEM',
    'Literature': 'Humanities',
    'History': 'Humanities',
    'Law': 'Humanities',
    'Art': 'Humanities',
    'Philosophy': 'Humanities',
    'Economy': 'Social_sciences',
    'Management': 'Social_sciences',
    'Education': 'Social_sciences',
}

from mgtbench.utils import setup_seed

def process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def process_text_truthfulqa_adv(text):

    if "I am sorry" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    if "as an AI language model" in text or "As an AI language model" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    return text


def check_period(texts):
    for i in range(len(texts)):
        if texts[i][-1] != ".":
            texts[i] += "."
    return texts


def load_TruthfulQA(detectLLM):
    f = pd.read_csv("datasets/TruthfulQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['Best Answer'].tolist()
    a_human = check_period(a_human)
    print(a_human)
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()
    c = f['Category'].tolist()

    res = []
    for i in range(len(q)):
        if len(
            a_human[i].split()) > 1 and len(
            a_chat[i].split()) > 1 and len(
                a_chat[i]) < 2000:
            res.append([q[i], a_human[i], a_chat[i], c[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)

        data_new[data_partition]['category'].append(res[index_list[i]][3])
        data_new[data_partition]['category'].append(res[index_list[i]][3])

    return data_new


def load_SQuAD1(detectLLM):
    f = pd.read_csv("datasets/SQuAD1_LLMs.csv")
    q = f['Question'].tolist()
    a_human = [eval(_)['text'][0] for _ in f['answers'].tolist()]
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            a_human[i] = check_period(a_human[i])
            res.append([q[i], a_human[i], a_chat[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'

        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)
    return data_new


def load_NarrativeQA(detectLLM):
    f = pd.read_csv("datasets/NarrativeQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['answers'].tolist()
    a_human = [_.split(";")[0] for _ in a_human]
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(
            a_human[i].split()) > 1 and len(
            a_chat[i].split()) > 1 and len(
            a_human[i].split()) < 150 and len(
                a_chat[i].split()) < 150:
            a_human[i] = check_period(a_human[i])
            res.append([q[i], a_human[i], a_chat[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)
    return data_new


def load(name, detectLLM, category='Art', seed=0):
    if name in ['TruthfulQA', 'SQuAD1', 'NarrativeQA']:
        load_fn = globals()[f'load_{name}']
        return load_fn(detectLLM)
    elif name in ["Essay", "Reuters", "WP"]:
        data = load_old_data(name, detectLLM)
        return data    
    elif name == 'AITextDetect':
        if category in CATEGORIES:
            data = load_subject_data(detectLLM, category, seed)
            return data    
        elif category in TOPICS:
            data = load_topic_data(detectLLM, topic=category, seed=seed)
            return data
        else:
            raise ValueError(f"Unknown category: {category}")


def load_old_data(name, detectLLM):
    f = pd.read_csv(f"data/{name}_LLMs.csv")
    a_human = f["human"].tolist()
    a_chat = f[f'{detectLLM}'].fillna("").tolist()

    res = []
    for i in range(len(a_human)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            res.append([a_human[i], a_chat[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data", disable=True):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][0]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(1)

    return data_new


def load_subject_data(detectLLM, category, seed=0):
    saved_data_path = f"/data_sda/zhiyuan/data_3407/{detectLLM}_{category}.json"
    if os.path.exists(saved_data_path):
        print('using saved data', saved_data_path)
        with open(saved_data_path, 'r') as f:
            data = json.load(f)
        train_machine_cnt = sum(data['train']['label'])
        train_human_cnt = len(data['train']['label']) - train_machine_cnt
        test_machine_cnt = sum(data['test']['label'])
        test_human_cnt = len(data['test']['label']) - test_machine_cnt
        print(f"train machine: {train_machine_cnt}, train human: {train_human_cnt}")
        print(f"test machine: {test_machine_cnt}, test human: {test_human_cnt}")
        return data

    print('loading human data')
    repo = "/data1/zzy/datasets/AI_Polish_clean"
    # repo = "AITextDetect/AI_Polish_clean"
    subject_human_data = load_dataset(repo, trust_remote_code=True, name='Human', split=category, cache_dir='/data1/zzy/cache/huggingface')

    print('loading machine data')
    mgt_data = load_dataset(repo, trust_remote_code=True, name=detectLLM, split=category, cache_dir='/data1/zzy/cache/huggingface')

    print('data loaded')

    # data mix up
    all_data = []
    smaller_len = min([len(subject_human_data), len(mgt_data)])

    subject_human_data = subject_human_data.shuffle(seed)
    for i in range(smaller_len): # 50:50
        all_data.append({'text': mgt_data[i]['text'], 'label': 1})
        all_data.append({'text': subject_human_data[i]['text'], 'label': 0})

    index_list = list(range(len(all_data)))
    random.shuffle(index_list)

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    total_num = len(all_data)
    for i in tqdm.tqdm(range(total_num), desc="parsing data", disable=False):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(all_data[index_list[i]]['text']))
        data_new[data_partition]['label'].append(all_data[index_list[i]]['label'])

    if not os.path.exists(saved_data_path):
        with open(saved_data_path, 'w') as f:
            json.dump(data_new, f)

    return data_new


def load_topic_data(detectLLM, topic, seed=0):
    setup_seed(seed)
    saved_data_path = f"/data_sda/zhiyuan/data_3407/{detectLLM}_{topic}.json"
    if os.path.exists(saved_data_path):
        print('using saved data', saved_data_path)
        with open(saved_data_path, 'r') as f:
            data = json.load(f)
        train_machine_cnt = sum(data['train']['label'])
        train_human_cnt = len(data['train']['label']) - train_machine_cnt
        test_machine_cnt = sum(data['test']['label'])
        test_human_cnt = len(data['test']['label']) - test_machine_cnt
        print(f"train machine: {train_machine_cnt}, train human: {train_human_cnt}")
        print(f"test machine: {test_machine_cnt}, test human: {test_human_cnt}")
        return data

    repo = "/data1/zzy/datasets/AI_Polish_clean"
    all_data = {}
    all_data['human'] = []
    all_data[topic] = []
    for subject in CATEGORIES:
        if TOPIC_MAPPING[subject] == topic:
            # repo = "AITextDetect/AI_Polish_clean"
            subject_human_data = load_dataset(repo, trust_remote_code=True, name='Human', split=subject, cache_dir='/data1/zzy/cache/huggingface')
            mgt_data = load_dataset(repo, trust_remote_code=True, name=detectLLM, split=subject, cache_dir='/data1/zzy/cache/huggingface')
            all_data['human'].append(subject_human_data)
            all_data[topic].append(mgt_data)
    
    # find the smallest length, balance between subjects and human/machine
    min_len = int(1e9)
    for i in range(len(all_data['human'])):
        min_len = min(min_len, len(all_data['human'][i]))
        min_len = min(min_len, len(all_data[topic][i]))

    # balance the data
    for i in range(len(all_data['human'])):
        all_data['human'][i] = all_data['human'][i].shuffle().select(range(min_len))
        all_data[topic][i] = all_data[topic][i].shuffle().select(range(min_len))

    # data mix up
    final_data = []
    for i in range(len(all_data['human'])):
        for j in range(min_len):
            final_data.append({'text': all_data[topic][i][j]['text'], 'label': 1})
            final_data.append({'text': all_data['human'][i][j]['text'], 'label': 0})

    index_list = list(range(len(final_data)))
    random.shuffle(index_list)

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    total_num = len(final_data)
    ratio = 0.7
    for i in tqdm.tqdm(range(total_num), desc="parsing data", disable=False):
        if i < total_num * ratio:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(final_data[index_list[i]]['text']))
        data_new[data_partition]['label'].append(final_data[index_list[i]]['label'])

    # if not os.path.exists(saved_data_path):
    with open(saved_data_path, 'w') as f:
        json.dump(data_new, f)

    return data_new



def download_data(model_name, category):
    return load_dataset(
        "AITextDetect/AI_Polish_clean",
        trust_remote_code=True,
        name=model_name,
        split=category,
        # cache_dir=cache_dir
    )

def prepare_attribution(category='Art', seed=0):
    setup_seed(seed)
    # human
    subject_human_data = load_dataset("AITextDetect/AI_Polish_clean", trust_remote_code=True, name='Human', split=category)

    # Prepare attribution data 
    model_data = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_data, m, category): m for m in MODELS}
        for future in futures:
            model_name = futures[future]
            try:
                model_data[model_name] = future.result()
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

    min_len = int(1e9)
    for m in MODELS:
        min_len = min(min_len, len(model_data[m]))

    # balance the data
    for m in MODELS:
        model_data[m] = model_data[m].shuffle().select(range(min_len))
        # print(m, len(model_data[m]))

    label_mapping = {'Human': 0, 'Moonshot': 1, 'gpt35': 2, 'Mixtral': 3, 'Llama3': 4, 'gpt-4omini': 5}

    all_data = []
    for m in MODELS:
        this_data = model_data[m]
        for d in this_data:
            all_data.append({'text': d['text'], 'label': label_mapping[m]})

    human_sample = subject_human_data.shuffle().select(range(min_len))
    for d in human_sample:
        all_data.append({'text': d['text'], 'label': label_mapping['Human']})

    data = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }
    }

    index_list = list(range(len(all_data)))
    random.shuffle(index_list)

    total_num = len(all_data)
    for i in tqdm.tqdm(range(total_num), desc="parsing data", disable=True):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data[data_partition]['text'].append(
            process_spaces(all_data[index_list[i]]['text']))
        data[data_partition]['label'].append(all_data[index_list[i]]['label'])

    return data


def prepare_attribution_topic(topic='STEM', seed=0):
    setup_seed(seed)
    # e.g. all_data['Moonshot']['STEM'] = [list of subject data in STEM]
    all_data = {} 
    # all_data[model] = [list of data in each subject], for the given topic
    all_data['human'] = []
    for detectLLM in MODELS:
        all_data[detectLLM] = []

    repo = "/data1/zzy/datasets/AI_Polish_clean"
    # repo = "AITextDetect/AI_Polish_clean"

    # load all subject data related to the topic
    for model in MODELS:
        for subject in CATEGORIES:
            if TOPIC_MAPPING[subject] != topic:
                continue
            mgt_data = load_dataset(repo, trust_remote_code=True, name=model, split=subject, cache_dir='/data1/zzy/cache/huggingface')
            all_data[model].append(mgt_data)
    
    for subject in CATEGORIES:
        if TOPIC_MAPPING[subject] != topic:
            continue
        subject_human_data = load_dataset(repo, trust_remote_code=True, name='Human', split=subject, cache_dir='/data1/zzy/cache/huggingface')
        all_data['human'].append(subject_human_data)

    # find the smallest length, balance between subjects
    min_len_dict = {} 
    for model in MODELS+['human']:
        min_len = int(1e9)
        for i in range(len(all_data[model])):
            min_len = min(min_len, len(all_data[model][i]))
        min_len_dict[model] = min_len

    # balance the data
    for model in MODELS+['human']:
        for i in range(len(all_data[model])):
            all_data[model][i] = all_data[model][i].shuffle().select(range(min_len_dict[model]))

    # now the topic has the same number of data, for each subject
    # balance the data between each models and human
    min_len = int(1e9)
    for model in MODELS+['human']:
        model_topic_total_num = 0
        for i in range(len(all_data[model])):
            model_topic_total_num += len(all_data[model][i])
        min_len = min(min_len, model_topic_total_num)

    label_mapping = {'human': 0, 'Moonshot': 1, 'gpt35': 2, 'Mixtral': 3, 'Llama3': 4, 'gpt-4omini': 5}
    # data mix up
    final_data = []
    for model in MODELS+['human']:
        cnt = 0
        idx = 0
        while cnt < min_len:
            for i in range(len(all_data[model])):
                final_data.append({'text': all_data[model][i][idx]['text'], 'label': label_mapping[model]})
                cnt += 1
            idx += 1

    data = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }
    }

    index_list = list(range(len(final_data)))
    random.shuffle(index_list)

    total_num = len(final_data)
    ratio = 0.8
    for i in tqdm.tqdm(range(total_num), desc="parsing data", disable=False):
        if i < total_num * ratio:
            data_partition = 'train'
        else:
            data_partition = 'test'
        data[data_partition]['text'].append(
            process_spaces(final_data[index_list[i]]['text']))
        data[data_partition]['label'].append(final_data[index_list[i]]['label'])

    return data
        

def load_attribution_topic(topic):
    assert topic in TOPICS
    saved_data_path = f"/data_sda/zhiyuan/data_3407/{topic}_attribution.json"
    if not os.path.exists("/data_sda/zhiyuan/data_3407/"):
        os.makedirs("/data_sda/zhiyuan/data_3407/")
    if not os.path.exists(saved_data_path):
        data = prepare_attribution_topic(topic, seed=3407)
        with open(saved_data_path, 'w') as f:
            json.dump(data, f)
    else:
        with open(saved_data_path, 'r') as f:
            data = json.load(f)

    return data

def load_attribution(category):
    saved_data_path = f"data/{category}_attribution_data.json"
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(saved_data_path):
        data = prepare_attribution(category, seed=3407)
        with open(saved_data_path, 'w') as f:
            json.dump(data, f)
    else:
        with open(saved_data_path, 'r') as f:
            data = json.load(f)

    return data


def prepare_incremental(order: list, category='Art', seed=0):
    '''
    Prepare incremental data for the given category containing specified models and human data,
    with the order of each model set given by the list of lists in "order".
    Example: order = [['Moonshot'], ['gpt35', 'Llama3']]
    '''
    setup_seed(seed)
    
    # Load human data
    subject_human_data = load_dataset(
        "AITextDetect/AI_Polish_clean", trust_remote_code=True, 
        name='Human', split=category
    )

    # Load models data
    model_data = {}
    all_models = [model for group in order for model in group]
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_data, m, category): m for m in all_models}
        for future in futures:
            model_name = futures[future]
            try:
                model_data[model_name] = future.result()
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

    # Determine the data length for each round
    if len(order[0]) == 1:
        first_round_len = len(model_data[order[0][0]])
    else:
        first_round_len = min(len(model_data[model]) for model in order[0])

    # Limit the human sample to the first round length
    human_sample = subject_human_data.shuffle().select(range(first_round_len))

    # Prepare model data length according to the first round's length
    for m in model_data:
        if len(model_data[m]) > first_round_len:
            model_data[m] = model_data[m].shuffle().select(range(first_round_len))

    # Train/test split
    split = 0.8
    train_data = {}
    test_data = {}
    for m in all_models:
        available_len = len(model_data[m])
        current_len = min(available_len, first_round_len)
        train_data[m] = model_data[m].select(range(int(current_len * split)))
        test_data[m] = model_data[m].select(range(int(current_len * split), current_len))

    train_data['Human'] = human_sample.select(range(int(first_round_len * split)))
    test_data['Human'] = human_sample.select(range(int(first_round_len * split), first_round_len))

    # Incremental data structure
    data = {
        'train': [],
        'test': []
    }
    for i, group in enumerate(order):
        # Train data: start with human data in the first round, add models as per the order list
        temp_train = {
            'text': [],
            'label': []
        }
        if i == 0:
            for d in train_data['Human']:
                temp_train['text'].append(d['text'])
                temp_train['label'].append(0)  # Human label

        for j, model_name in enumerate(group):
            model_len = min(len(train_data[model_name]), first_round_len) if i > 0 else len(train_data[model_name])
            model_train_data = train_data[model_name].select(range(model_len))
            
            for d in model_train_data:
                temp_train['text'].append(d['text'])
                temp_train['label'].append(j + 1 + sum(len(g) for g in order[:i]))  # Model-specific label

        if len(group) > 1 or i == 0:
            combined_data = list(zip(temp_train['text'], temp_train['label']))
            random.shuffle(combined_data)  # Shuffle when there are multiple models
            temp_train['text'], temp_train['label'] = zip(*combined_data)
        
        data['train'].append(temp_train)

        # Test data: include human data initially, add current and past models
        temp_test = {
            'text': [],
            'label': []
        }
        if i == 0:
            for d in test_data['Human']:
                temp_test['text'].append(d['text'])
                temp_test['label'].append(0)  # Human label

        for j, model_name in enumerate(group):
            model_len = min(len(test_data[model_name]), first_round_len) if i > 0 else len(test_data[model_name])
            model_test_data = test_data[model_name].select(range(model_len))
            
            for d in model_test_data:
                temp_test['text'].append(d['text'])
                temp_test['label'].append(j + 1 + sum(len(g) for g in order[:i]))  # Model-specific label

        if i > 0:
            # Include previous test data in the current round
            prev_test_data = data['test'][i - 1]
            temp_test['text'].extend(prev_test_data['text'])
            temp_test['label'].extend(prev_test_data['label'])

        data['test'].append(temp_test)

    return data

def load_incremental(order, category):
    saved_data_path = f"data/{category}_incremental_data.json"
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(saved_data_path):
        data = prepare_incremental(order, category, seed=3407)
        with open(saved_data_path, 'w') as f:
            json.dump(data, f)
    else:
        with open(saved_data_path, 'r') as f:
            data = json.load(f)

    return data