import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import torch
import csv
import re

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_topic_data
from mgtbench.utils import setup_seed
from mgtbench.auto import DetectOutput


TOPICS = ['STEM', 'Humanities', 'Social_sciences']

llms = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']


def get_demo_data(data, train_size, test_size):
    demo = {}
    demo['train'] = {'text': data['train']['text'][:train_size], 'label': data['train']['label'][:train_size]}
    demo['test'] = {'text': data['test']['text'][:test_size], 'label': data['test']['label'][:test_size]}
    return demo

checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
# Function to get the largest checkpoint directory
def get_path(folder):
    
    if os.path.exists(folder):
        # Find all subdirectories matching the checkpoint pattern
        checkpoint_dirs = [d for d in os.listdir(folder) if checkpoint_pattern.match(d)]
        
        if checkpoint_dirs:
            # Get the full paths of the checkpoint directories
            checkpoint_paths = [os.path.join(folder, d) for d in checkpoint_dirs]
            # Find the directory with the latest modification time
            latest_checkpoint_path = max(checkpoint_paths, key=os.path.getmtime)
            print(f'Loading {latest_checkpoint_path}')
        else:
            print(f"No checkpoints found in {folder}")
            latest_checkpoint_path = None
    else:
        print(f"Directory does not exist: {folder}")
        latest_checkpoint_path = None
    
    return latest_checkpoint_path    


def log_domain_result(csv_file: str, result: DetectOutput, detectLLM: str, source_topic: str, target_topic: str, base_model='None'):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Base_model', 'Detect_LLM', 'Source_topic', 'Target_topic',
                                 'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])
    else:
        with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([base_model, detectLLM, source_topic, target_topic, 
                                round(result.test.acc, 4), round(result.test.precision, 4), round(result.test.recall, 4), round(result.test.f1, 4), round(result.test.auc, 4)])


def log_llm_result(csv_file: str, result: DetectOutput, source_llm: str, target_llm: str, source_topic: str, base_model='None'):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Base_model', 'Source_LLM', 'Target_LLM', 'Source_topic',
                                 'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])
    else:
        with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([base_model, source_llm, target_llm, source_topic, 
                                round(result.test.acc, 4), round(result.test.precision, 4), round(result.test.recall, 4), round(result.test.f1, 4), round(result.test.auc, 4)])


def transfer_llm(source_topic, source_llm, target_llm, base_model, result_csv):
    setup_seed(3407)
    data_target = load_topic_data(topic=source_topic, detectLLM=target_llm)
    data_target = get_demo_data(data_target, 100, 2000)

    # change the model path
    model_path = f'/data_sde/zhiyuan/topic_models/{base_model}_{source_llm}_{source_topic}_3_64'
    actual_model_path = get_path(model_path)
    method = AutoDetector.from_detector_name('LM-D', model_name_or_path=actual_model_path, tokenizer_path=actual_model_path)
    experiment = AutoExperiment.from_experiment_name('supervised',detector=[method])
    experiment.load_data(data_target)

    print('----------')
    print('Source topic:', source_topic)
    print('Target topic:', target_topic)
    res = experiment.launch(need_finetune=False,
                            eval=True)
    print('----------')
    print(res[0].train)
    print(res[0].test)
    print('----------')
    log_llm_result(result_csv, res[0], source_llm, target_llm, source_topic, base_model)


def transfer_domain(source_topic, target_topic, detectLLM, base_model, result_csv):
    setup_seed(3407)
    data_target = load_topic_data(topic=target_topic, detectLLM=detectLLM)
    data_target = get_demo_data(data_target, 100, 2000)

    # change the model path
    model_path = f'/data_sde/zhiyuan/topic_models/{base_model}_{detectLLM}_{source_topic}_3_64'
    actual_model_path = get_path(model_path)
    method = AutoDetector.from_detector_name('LM-D', model_name_or_path=actual_model_path, tokenizer_path=actual_model_path)
    experiment = AutoExperiment.from_experiment_name('supervised',detector=[method])
    experiment.load_data(data_target)

    print('----------')
    print('Source topic:', source_topic)
    print('Target topic:', target_topic)
    res = experiment.launch(need_finetune=False,
                            eval=True)
    print('----------')
    print(res[0].train)
    print(res[0].test)
    print('----------')
    log_domain_result(result_csv, res[0], detectLLM, source_topic, target_topic, base_model)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=['domain', 'llm'])
    parser.add_argument('--source_topic', type=str, choices=TOPICS)
    parser.add_argument('--target_topic', type=str, choices=TOPICS)
    parser.add_argument('--detectLLM', type=str, choices=llms)
    parser.add_argument('--base_model', type=str)
    parser.add_argument('--source_llm', type=str, choices=llms)
    parser.add_argument('--target_llm', type=str, choices=llms)
    parser.add_argument('--result_csv', type=str)

    args = parser.parse_args()
    source_topic = args.source_topic
    target_topic = args.target_topic
    detectLLM = args.detectLLM
    base_model = args.base_model
    source_llm = args.source_llm
    target_llm = args.target_llm
    result_csv = args.result_csv

    if args.task == 'llm':
        transfer_llm(
                    source_topic=source_topic,
                    source_llm=source_llm,
                    target_llm=target_llm,
                    base_model=base_model,
                    result_csv=result_csv
                    )
    else:
        transfer_domain(
                        source_topic=source_topic,
                        target_topic=target_topic,
                        detectLLM=detectLLM,
                        base_model=base_model,
                        result_csv=result_csv
                        )
