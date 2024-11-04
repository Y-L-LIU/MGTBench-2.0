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
    Prepare incremental data for the given category containing 6 models and human data,
    the order of each model is given by the order list
    order: ['Moonshot', 'gpt35', 'Llama3'] for example
    '''
    setup_seed(seed)
    # human data
    subject_human_data = datasets.load_dataset("AITextDetect/AI_Polish_clean", trust_remote_code=True, name='Human', split=category, cache_dir='/data1/zzy/cache/huggingface')

    # models data
    model_data = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_data, m, category, '/data1/zzy/cache/huggingface'): m for m in MODELS}
        for future in futures:
            model_name = futures[future]
            try:
                model_data[model_name] = future.result()
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

    # TODO: now contrain the length of data to be the same
    min_len = int(1e9)
    for m in MODELS:
        min_len = min(min_len, len(model_data[m]))

    for m in MODELS:
        model_data[m] = model_data[m].shuffle().select(range(min_len))

    human_sample = subject_human_data.shuffle().select(range(min_len))

    split = 0.8
    train_data = {}
    test_data = {}
    for m in MODELS:
        train_data[m] = model_data[m].select(range(int(min_len * split)))
        test_data[m] = model_data[m].select(range(int(min_len * split), min_len))

    train_data['Human'] = human_sample.select(range(int(min_len * split)))
    test_data['Human'] = human_sample.select(range(int(min_len * split), min_len))

    # label_mapping = {'Human': 0, 'Moonshot': 1, 'gpt35': 2, 'Mixtral': 3, 'Llama3': 4, 'gpt-4omini': 5}
    num_class = len(order)

    # a list of data, len(train) == len(order), for each round of incremental learning
    data = {
        'train': [],
        'test': []
    }
    for i in range(num_class):
        if i == 0:
            # for train data, first data should contain shuffled human AND one model data
            this_model_data = train_data[order[i]]
            temp = {
                'text': [],
                'label': []
            }
            data_list = []
            for d in this_model_data:
                data_list.append({'text': d['text'], 'label': i + 1})
            # TODO: here all human data is added to the first data, maybe we can add human data to each stage (as mitigation)
            for d in train_data['Human']:
                data_list.append({'text': d['text'], 'label': 0})

            random.shuffle(data_list)
            for d in data_list:
                temp['text'].append(d['text'])
                temp['label'].append(d['label'])

            data['train'].append(temp)
            # for test data: no need to shuffle
            temp = {
                'text': [],
                'label': []
            }
            for d in test_data[order[i]]:
                temp['text'].append(d['text'])
                temp['label'].append(i + 1)
            for d in test_data['Human']:
                temp['text'].append(d['text'])
                temp['label'].append(0)

            data['test'].append(temp)
        else:
            # for train: the rest of the data contains only one model data
            this_model_data = train_data[order[i]]
            temp = {
                'text': [],
                'label': []
            }
            for d in this_model_data:
                temp['text'].append(d['text'])
                temp['label'].append(i + 1)

            data['train'].append(temp)

            # for test: each stage contains human data, and the previous model data
            temp = {
                'text': [],
                'label': []
            }
            prev_test_data = data['test'][i - 1] # i > 0
            temp = copy.deepcopy(prev_test_data)
            # add current stage model data, i.e. a new class
            for d in test_data[order[i]]:
                temp['text'].append(d['text'])
                temp['label'].append(i + 1)

            data['test'].append(temp)

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