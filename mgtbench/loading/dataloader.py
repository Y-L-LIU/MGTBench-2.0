import random
import datasets
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
    
    elif name == 'AITextDetect':
        subject_human_data = datasets.load_dataset("AITextDetect/AI_Polish_clean", trust_remote_code=True, name='Human', split=category)
        mgt_data = datasets.load_dataset("AITextDetect/AI_Polish_clean", trust_remote_code=True, name=detectLLM, split=category)

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
        for i in tqdm.tqdm(range(total_num), desc="parsing data", disable=True):
            if i < total_num * 0.8:
                data_partition = 'train'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(all_data[index_list[i]]['text']))
            data_new[data_partition]['label'].append(all_data[index_list[i]]['label'])
        return data_new

    else:
        raise ValueError(f'Unknown dataset {name}')


def download_data(model_name, category):
    return datasets.load_dataset(
        "AITextDetect/AI_Polish_clean",
        trust_remote_code=True,
        name=model_name,
        split=category,
        # cache_dir=cache_dir
    )

def prepare_attribution(category='Art', seed=0):
    setup_seed(seed)
    # human
    subject_human_data = datasets.load_dataset("AITextDetect/AI_Polish_clean", trust_remote_code=True, name='Human', split=category)

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
    subject_human_data = datasets.load_dataset(
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