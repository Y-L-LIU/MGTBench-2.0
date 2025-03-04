import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import torch
import csv
import re

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_topic_data
from mgtbench.utils import setup_seed
from mgtbench.auto import DetectOutput, Metric

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

LLMS = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']

TEST_SIZE = 2000

# change to your path
mitigate_save_dir = '/data_sda/zhiyuan/transfer_mitigate_topics'
domain_result_csv = './results/transfer_mitigate/transfer_domain_mitigate_topic.csv'
llm_result_csv = './results/transfer_mitigate/transfer_llm_mitigate_topic.csv'

# Regular expression pattern to match "checkpoint-{num}"
checkpoint_pattern = re.compile(r'checkpoint-(\d+)')

base_dir = '/data_sda/zhiyuan/topic_models'  # Update this to your base folder path, with trained models

# Function to get the largest checkpoint directory
def get_path(model, llm, topic):
    dir_path = os.path.join(base_dir, f"{model}_{llm}_{topic}_3_64") # Update this to your trained model path
    if os.path.exists(dir_path):
        # Find all subdirectories matching the checkpoint pattern
        checkpoint_dirs = [d for d in os.listdir(dir_path) if checkpoint_pattern.match(d)]
        if checkpoint_dirs:
            # Get the full paths of the checkpoint directories
            checkpoint_paths = [os.path.join(dir_path, d) for d in checkpoint_dirs]
            # Find the directory with the latest modification time
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


def log_domain_result(csv_file: str, 
                      train_result: DetectOutput,
                      test_result: DetectOutput,
                      detectLLM: str, 
                      source_topic: str, 
                      target_topic: str, 
                      base_model: str,
                      mitigate_size: int,
                      epoch: int,
                      batch_size: int
                      ):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Base_model', 'Detect_LLM', 'Source_topic', 'Target_topic',
                                'Mitigate_size', 'Epoch', 'Batch_size',
                                'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc',
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([base_model, detectLLM, source_topic, target_topic, 
                            mitigate_size, epoch, batch_size,
                            round(train_result.acc, 4), round(train_result.precision, 4), round(train_result.recall, 4), round(train_result.f1, 4), round(train_result.auc, 4),
                            round(test_result.acc, 4), round(test_result.precision, 4), round(test_result.recall, 4), round(test_result.f1, 4), round(test_result.auc, 4)])
    

def log_llm_result( csv_file: str,
                    train_result: DetectOutput,
                    test_result: DetectOutput,
                    source_llm: str,
                    target_llm: str,
                    source_topic: str,
                    base_model: str,
                    mitigate_size: int,
                    epoch: int,
                    batch_size: int
                    ):
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Base_model', 'Source_LLM', 'Target_LLM', 'Source_topic',
                                'Mitigate_size', 'Epoch', 'Batch_size',
                                'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc',
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([base_model, source_llm, target_llm, source_topic, 
                            mitigate_size, epoch, batch_size,
                            round(train_result.acc, 4), round(train_result.precision, 4), round(train_result.recall, 4), round(train_result.f1, 4), round(train_result.auc, 4),
                            round(test_result.acc, 4), round(test_result.precision, 4), round(test_result.recall, 4), round(test_result.f1, 4), round(test_result.auc, 4)])


def transfer_domain(result_csv, base_model, source_topic, target_topic, detectLLM):
    data_target = load_topic_data(detectLLM=detectLLM, topic=target_topic)
    trained_model_path = get_path(base_model, detectLLM, source_topic)

    if source_topic == target_topic:
        detector = AutoDetector.from_detector_name('LM-D', model_name_or_path=trained_model_path, tokenizer_path=trained_model_path)
        experiment = AutoExperiment.from_experiment_name('supervised',detector=[detector])
        exp_data = get_demo_data(data_target, 10, TEST_SIZE)
        experiment.load_data(exp_data)
        print('----------')
        print('DetectLLM:', detectLLM)
        print('Source Category:', source_topic)
        print('----------')
        print('Target Category:', target_topic)
        res = experiment.launch(need_finetune=False)
        print('----------')
        print(res[0].train)
        print(res[0].test)
        print('----------')
        log_domain_result(result_csv, res[0].train, res[0].test, detectLLM, source_topic, target_topic, base_model, 0, None, None)
        return

    data_sizes  = [0, 100, 200, 300, 500, 800]
    print('----------')
    print('DetectLLM:', detectLLM)
    print('Source Category:', source_topic)
    print('Target Category:', target_topic)    
    print('----------')
    for size in data_sizes:
        mitigate_data = get_demo_data(data_target, size, TEST_SIZE)

        print('Mitigate Size:', size)
        torch.cuda.empty_cache()

        # TODO: run epoch 2 if necessary
        for epoch in [1]:
            for batch_size in [32]:
                model_save_dir = f"{mitigate_save_dir}/domain/{base_model}/LM-D_{detectLLM}_{source_topic}_to_{target_topic}_mitigate_{size}_epoch_{epoch}_batch_{batch_size}"
                config = {
                    'need_finetune': True,
                    'save_path': model_save_dir,
                    'epochs': 1,
                    'batch_size': batch_size,
                    'disable_tqdm': True
                }
                if size == 0:
                    if epoch == 2 or batch_size == 64:
                        break
                    config['need_finetune'] = False
                    config['eval'] = True

                detector = AutoDetector.from_detector_name('LM-D', model_name_or_path=trained_model_path, tokenizer_path=trained_model_path)
                experiment = AutoExperiment.from_experiment_name('supervised',detector=[detector])
                experiment.load_data(mitigate_data)
                res = experiment.launch(**config)
                print('----------')
                # print(res[0].train)
                print(res[0].test)
                # print('----------')
                if res[0].train is None:
                    res[0].train = Metric(0, 0, 0, 0, 0)
                log_domain_result(result_csv, res[0].train, res[0].test, detectLLM, source_topic, target_topic, base_model, size, epoch, batch_size)
                torch.cuda.empty_cache()


def transfer_llm(result_csv, base_model, source_topic, source_llm, target_llm):
    target_data = load_topic_data(detectLLM=target_llm, topic=source_topic)
    trained_model_path = get_path(base_model, source_llm, source_topic)

    if target_llm == source_llm:
        detector = AutoDetector.from_detector_name('LM-D', model_name_or_path=trained_model_path, tokenizer_path=trained_model_path)
        experiment = AutoExperiment.from_experiment_name('supervised', detector=[detector])
        exp_data = get_demo_data(target_data, 10, TEST_SIZE)
        experiment.load_data(exp_data)
        print('----------')
        print('Model:', base_model)
        print('Category:', source_topic)
        print('Source DetectLLM:', source_llm)
        print('----------')
        torch.cuda.empty_cache()
        res = experiment.launch(need_finetune=False)
        print('----------')
        print(res[0].test)
        print('----------')
        log_llm_result(result_csv, res[0].train, res[0].test, source_llm, target_llm, source_topic, base_model, 0, None, None)
        return
    
    else:
        # TODO: use other sizes if necessary
        mitigate_sizes = [0, 100, 200, 300, 500, 800]
        print('----------')
        print('Model:', base_model)
        print('Category:', source_topic)
        print('Source DetectLLM:', source_llm)
        for size in mitigate_sizes:
            mitigate_data = get_demo_data(target_data, size, TEST_SIZE)

            torch.cuda.empty_cache()
            # TODO: run epoch 2 if necessary
            for epoch in [1]:
                for batch_size in [32]:
                    model_save_dir = f"{mitigate_save_dir}/llm/{base_model}/LM-D_{source_topic}_{source_llm}_to_{target_llm}_mitigate_{size}_epoch_{epoch}_batch_{batch_size}"
                    config = {
                        'need_finetune': True,
                        'save_path': model_save_dir,
                        'epochs': epoch,
                        'batch_size': batch_size,
                        'disable_tqdm': True
                    }
                    if size == 0: 
                        config['need_finetune'] = False
                        config['eval'] = True

                    detector = AutoDetector.from_detector_name('LM-D', model_name_or_path=trained_model_path, tokenizer_path=trained_model_path)
                    # Create an experiment for the given detector
                    experiment = AutoExperiment.from_experiment_name('supervised', detector=[detector])

                    experiment.load_data(mitigate_data)
                    res = experiment.launch(**config)
                    print('----------')
                    print(res[0].test)
                    print('----------')
                    if res[0].train is None:
                        res[0].train = Metric(0, 0, 0, 0, 0)
                    log_llm_result(result_csv, res[0].train, res[0].test, source_llm, target_llm, source_topic, base_model, size, epoch, batch_size)
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=['domain', 'llm'])
    parser.add_argument('--source_topic', type=str, choices=TOPICS)
    parser.add_argument('--target_topic', type=str, choices=TOPICS)
    parser.add_argument('--detectLLM', type=str, choices=LLMS)
    parser.add_argument('--base_model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--source_llm', type=str, choices=LLMS)
    parser.add_argument('--target_llm', type=str, choices=LLMS)

    args = parser.parse_args()
    task = args.task
    source_topic = args.source_topic
    target_topic = args.target_topic
    detectLLM = args.detectLLM
    base_model = args.base_model
    source_llm = args.source_llm
    target_llm = args.target_llm

    setup_seed(3407)
    if args.task == 'llm':
        assert base_model in ['distilbert', 'roberta-base']
        assert source_topic in TOPICS
        assert source_llm in LLMS
        assert target_llm in LLMS
        transfer_llm(base_model=base_model,
                    source_topic=source_topic,
                    source_llm=source_llm,
                    target_llm=target_llm,
                    result_csv=llm_result_csv
                    )
       
    elif args.task == 'domain':
        assert base_model in ['distilbert', 'roberta-base']
        assert source_topic in TOPICS
        assert target_topic in TOPICS
        assert detectLLM in LLMS
        transfer_domain(base_model=base_model,
                        source_topic=source_topic,
                        target_topic=target_topic,
                        detectLLM=detectLLM,
                        result_csv=domain_result_csv
                        )
