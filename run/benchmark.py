import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch
import argparse
import csv
import re
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed
from mgtbench.auto import DetectOutput

CATEGORIES = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry'
            ]

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

LLMS = ['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini']

# change to your local model path
roberta_path = '/data1/zzy/roberta-base'
distilbert_path = '/data1/models/distilbert-base-uncased'
gpt2 = '/data1/zzy/gpt2-medium'
# fast-detectGPT
gpt_neo = '/data_sda/zhiyuan/models/gpt-neo-2.7B'
gpt_j = '/data_sda/zhiyuan/models/gpt-j-6B'
# binoculars
falcon_7b = '/data_sda/zhiyuan/models/falcon-7b'
falcon_7b_instruct = '/data_sda/zhiyuan/models/falcon-7b-instruct'

llama2_chat = '/data1/models/Llama-2-7b-chat-hf'

# change to your local save path
save_dir = '/data_sda/zhiyuan/topic_models'
base_dir = save_dir

def log_result_lmd(csv_file: str, result: DetectOutput, method: str, detectLLM: str, category: str, base_model, epoch: int, batch_size: int):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['method', 'detectLLM', 'category', 'base_model', 'epoch', 'batch_size',
                                'criterion', 'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc', 
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        criterion = result.name if result.name else 'None'
        csvwriter.writerow([method, detectLLM, category, base_model, epoch, batch_size, criterion,
                             round(result.train.acc, 4), round(result.train.precision, 4), round(result.train.recall, 4), round(result.train.f1, 4), round(result.train.auc, 4),
                             round(result.test.acc, 4), round(result.test.precision, 4), round(result.test.recall, 4), round(result.test.f1, 4), round(result.test.auc, 4)])

def log_result(csv_file: str, result: DetectOutput, method: str, detectLLM: str, category: str, base_model='None'):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['method', 'detectLLM', 'category', 'base_model', 'criterion', 'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc', 
                                            'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        criterion = result.name if result.name else 'None'
        csvwriter.writerow([method, detectLLM, category, base_model, criterion,
                             round(result.train.acc, 4), round(result.train.precision, 4), round(result.train.recall, 4), round(result.train.f1, 4), round(result.train.auc, 4),
                             round(result.test.acc, 4), round(result.test.precision, 4), round(result.test.recall, 4), round(result.test.f1, 4), round(result.test.auc, 4)])

# to get the trained LM-D model
checkpoint_pattern = re.compile(r'checkpoint-(\d+)')

def get_path(model, llm, topic, epoch, batch_size):
    dir_path = os.path.join(base_dir, f"{model}_{llm}_{topic}_{epoch}_{batch_size}")
    
    if os.path.exists(dir_path):
        checkpoint_dirs = [d for d in os.listdir(dir_path) if checkpoint_pattern.match(d)]
        if checkpoint_dirs:
            checkpoint_paths = [os.path.join(dir_path, d) for d in checkpoint_dirs]
            latest_checkpoint_path = max(checkpoint_paths, key=os.path.getmtime)
            print(f'Loading {latest_checkpoint_path}')
        else:
            print(f"No checkpoints found in {dir_path}")
            latest_checkpoint_path = None
    else:
        print(f"Directory does not exist: {dir_path}")
        latest_checkpoint_path = None
    
    return latest_checkpoint_path    


def get_demo_data(data, train_size, test_size):
    demo = {}
    demo['train'] = {'text': data['train']['text'][:train_size], 'label': data['train']['label'][:train_size]}
    demo['test'] = {'text': data['test']['text'][:test_size], 'label': data['test']['label'][:test_size]}
    return demo

def experiment(csv_file, method, detectLLM):
    setup_seed(3407)

    # metric-based, threshold
    if method in ['ll', 'rank', 'LRR', 'rank_GLTR', 'entropy']:
        # base_models = [gpt2, llama2_chat]
        base_models = [llama2_chat]
        for base_model in base_models:
            detector = AutoDetector.from_detector_name(method, model_name_or_path=base_model)
            experiment = AutoExperiment.from_experiment_name('threshold',detector=[detector])
            for topic in TOPICS:
                data = load('AITextDetect', detectLLM=detectLLM, category=topic)
                data = get_demo_data(data,train_size=1000, test_size=2000)
                experiment.load_data(data)
                res = experiment.launch()
                print('==========')
                print('train:', res[0].train)
                print('test:', res[0].test)
                log_result(csv_file, res[0], method, detectLLM, topic, base_model=base_model)
                if len(res) > 1:
                    print('==========')
                    print('train:', res[1].train)
                    print('test:', res[1].test)
                    log_result(csv_file, res[1], method, detectLLM, topic, base_model=base_model)
                torch.cuda.empty_cache()

    # model-based
    elif method in ['LM-D']:
        for topic in TOPICS:
            data = load('AITextDetect', detectLLM=detectLLM, category=topic)
            # for model_name in ['distilbert', 'roberta-base']:
            for model_name in ['distilbert']:
                epoch = 3
                batch_size = 64
                path = get_path(model_name, detectLLM, topic, epoch, batch_size) # for LM-D
                if path is None:
                    model_path = distilbert_path if model_name == 'distilbert' else roberta_path
                    MAX = 10000
                    data = get_demo_data(data,train_size=MAX, test_size=2000)
                    # train the model
                    detector = AutoDetector.from_detector_name(method, model_name_or_path=model_path, tokenizer_path=model_path)
                    experiment = AutoExperiment.from_experiment_name('supervised', detector=[detector])
                    experiment.load_data(data)
                    model_save_dir = f"{save_dir}/{model_name}_{detectLLM}_{topic}_{epoch}_{batch_size}"

                    config = {
                        'need_finetune': True,
                        'save_path': model_save_dir,
                        'epochs': epoch,
                        'batch_size': batch_size,
                        'disable_tqdm': True,
                        }
                    res = experiment.launch(**config)

                    print('==========')
                    print('train:', res[0].train)
                    print('test:', res[0].test)
                    log_result(csv_file, res[0], method, detectLLM, topic, base_model=model_name)
                    torch.cuda.empty_cache()
                else:
                    data = load('AITextDetect', detectLLM=detectLLM, category=topic)
                    data = get_demo_data(data,train_size=1000, test_size=2000)
                    detector = AutoDetector.from_detector_name(method, model_name_or_path=path, tokenizer_path=path)
                    experiment = AutoExperiment.from_experiment_name('supervised', detector=[detector])
                    experiment.load_data(data)
                    res = experiment.launch(need_finetune=False)
                    print('==========')
                    print('train:', res[0].train)
                    print('test:', res[0].test)
                    log_result(csv_file, res[0], method, detectLLM, topic, base_model=model_name)
                    torch.cuda.empty_cache()

    elif method in ['RADAR', 'chatgpt-detector']:
        for topic in TOPICS:
            data = load('AITextDetect', detectLLM=detectLLM, category=topic)
            data = get_demo_data(data,train_size=1000, test_size=2000)
            if method == 'RADAR': # https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B
                path = '/data_sda/zhiyuan/models/Radar'
            elif method == 'chatgpt-detector': # https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta
                path = '/data_sda/zhiyuan/models/chatgpt-detector-roberta' 
            detector = AutoDetector.from_detector_name(method, model_name_or_path=path, tokenizer_path=path)
            experiment = AutoExperiment.from_experiment_name('supervised', detector=[detector])
            experiment.load_data(data)
            res = experiment.launch()
            print('==========')
            print('train:', res[0].train)
            print('test:', res[0].test)
            log_result(csv_file, res[0], method, detectLLM, topic)
            torch.cuda.empty_cache()

    # perturbation-based
    elif method in ['fast-detectGPT']: # 'detectGPT', 'NPR', 'DNA-GPT' are too slow
        scoring_model_name_or_path = gpt_neo
        reference_model_name_or_path = gpt_j
        detector = AutoDetector.from_detector_name('fast-detectGPT', 
                                                    scoring_model_name_or_path=scoring_model_name_or_path,
                                                    reference_model_name_or_path= reference_model_name_or_path
                                                    )
        experiment = AutoExperiment.from_experiment_name('perturb', detector=[detector])

        for topic in TOPICS:
            data = load('AITextDetect', detectLLM=detectLLM, category=topic)
            data = get_demo_data(data,train_size=1000, test_size=2000)
            experiment.load_data(data)
            res = experiment.launch()
            print('==========')
            print(res[0])
            print('train:', res[0].train)
            print('test:', res[0].test)
            log_result(csv_file, res[0], method, detectLLM, topic)
            if len(res) > 1:
                print('==========')
                print(res[1])
                print('train:', res[1].train)
                print('test:', res[1].test)
                log_result(csv_file, res[1], method, detectLLM, topic)
            torch.cuda.empty_cache()
        
    # binoculars
    elif method in ['Binoculars']:
        observer_model_name_or_path = falcon_7b
        performer_model_name_or_path = falcon_7b_instruct

        detector = AutoDetector.from_detector_name('Binoculars', 
                                                    observer_model_name_or_path=observer_model_name_or_path,
                                                    performer_model_name_or_path= performer_model_name_or_path,
                                                    max_length=1024,
                                                    mode='accuracy', # accuracy (f1)
                                                    threshold='new'
                                                    )
        experiment = AutoExperiment.from_experiment_name('threshold', detector=[detector])
        for topic in TOPICS:
            data = load('AITextDetect', detectLLM=detectLLM, category=topic)
            data = get_demo_data(data,train_size=1000, test_size=2000)
            experiment.load_data(data)
            res = experiment.launch()
            print('==========')
            print('train:', res[0].train)
            print('test:', res[0].test)
            log_result(csv_file, res[0], method, detectLLM, topic)
            if len(res) > 1:
                print('==========')
                print('train:', res[1].train)
                print('test:', res[1].test)
                log_result(csv_file, res[1], method, detectLLM, topic)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--method', type=str, choices=['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR', 'Binoculars',
                                                        'detectGPT', 'NPR', 'DNA-GPT', 'fast-detectGPT',
                                                        'LM-D', 'RADAR', 'chatgpt-detector',
                                                        'demasq'])
    parser.add_argument('--detectLLM', type=str)

    args = parser.parse_args()
    csv_file = args.csv_path
    method = args.method
    detectLLM = args.detectLLM

    experiment(csv_file, method, detectLLM)
