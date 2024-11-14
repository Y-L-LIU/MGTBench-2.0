import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import torch
import csv
import re

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed

category = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry']

llms = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3']

distilbert = '/data1/models/distilbert-base-uncased'

base_dir = '/data1/lyl/mgtout'  # Update this to your base folder path, with trained models

mitigate_save_dir = '/data_sda/zhiyuan/transfer_mitigate'
domain_result_csv = 'transfer_domain_mitigate.csv'
llm_result_csv = 'transfer_llm_mitigate.csv'

# Regular expression pattern to match "checkpoint-{num}"
checkpoint_pattern = re.compile(r'checkpoint-(\d+)')


# Function to get the largest checkpoint directory
def get_path(model, llm, category):
    dir_path = os.path.join(base_dir, f"{model}/LM-D_{llm}_{category}")
    
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



def transfer_domain(base_model, source_subject, target_subject, detectLLM):
    data_target = load('AITextDetect', detectLLM=detectLLM, category=target_subject)
    path = get_path(base_model, detectLLM, source_subject)

    if source_subject == target_subject:
        metric2 = AutoDetector.from_detector_name('LM-D', model_name_or_path=path, tokenizer_path=path)
        experiment2 = AutoExperiment.from_experiment_name('supervised',detector=[metric2])
        experiment2.load_data(data_target)
        print('----------')
        print('DetectLLM:', detectLLM)
        print('Source Category:', source_subject)
        print('----------')
        print('Target Category:', target_subject)
        res = experiment2.launch(need_finetune=False)
        print('----------')
        print(res[0].train)
        print(res[0].test)
        print('----------')

        if not os.path.exists(domain_result_csv):
            with open(domain_result_csv, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Model', 'DetectLLM', 'Source_Category', 'Target_Category', 'Mitigate Size', 'Epoch', 'Train F1', 'Test F1'])
        else:
            with open(domain_result_csv, 'a', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([base_model, detectLLM, source_subject, target_subject, 0, 0, round(res[0].train.f1, 4), round(res[0].test.f1, 4)])
            
        return

    # TODO: use small data for fine-tuning
    data_sizes  = [100, 200, 500, 800]
    
    for size in data_sizes:
        mitigate_data = {}
        mitigate_data['train'] = {}
        mitigate_data['train']['text'] = data_target['train']['text'][:size]
        mitigate_data['train']['label'] = data_target['train']['label'][:size]
        mitigate_data['test'] = data_target['test']

        # prev_size = size

        print('----------')
        print('DetectLLM:', detectLLM)
        print('Source Category:', source_subject)
        print('----------')
        print('Target Category:', target_subject)
        print('Mitigate Size:', size)
        torch.cuda.empty_cache()

        model_save_dir = f"{mitigate_save_dir}/domain/{base_model}/LM-D_{detectLLM}_{source_subject}_to_{target_subject}_mitigate_{size}"
        # TODO: run epoch 2 if necessary
        for epoch in [1]:
            config = {
                'need_finetune': True,
                'save_path': model_save_dir,
                'epochs': 1,
                'batch_size': 32,
                'disable_tqdm': True
            }
            if size == 0:
                if epoch == 2:
                    break
                config['need_finetune'] = False
                config['eval'] = True

            metric2 = AutoDetector.from_detector_name('LM-D', model_name_or_path=path, tokenizer_path=path)
            experiment2 = AutoExperiment.from_experiment_name('supervised',detector=[metric2])
            experiment2.load_data(mitigate_data)
            res = experiment2.launch(**config)
            print('----------')
            print(res[0].train)
            print(res[0].test)
            print('----------')

            if not os.path.exists(domain_result_csv):
                with open(domain_result_csv, 'w', newline='', encoding='utf-8') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Model', 'DetectLLM', 'Source_Category', 'Target_Category', 'Mitigate Size', 'Epoch', 'Train F1', 'Test F1'])

            with open(domain_result_csv, 'a', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                size = 0 if size == 1 else size
                csvwriter.writerow([base_model, detectLLM, source_subject, target_subject, size, epoch, round(res[0].train.f1, 4), round(res[0].test.f1, 4)])

            torch.cuda.empty_cache()


def transfer_llm(base_model, source_subject, source_llm, target_llm):
    target_data = load('AITextDetect', detectLLM=target_llm, category=source_subject)
    path = get_path(base_model, source_llm, source_subject)

    if target_llm == source_llm:
        metric = AutoDetector.from_detector_name('LM-D', model_name_or_path=path, tokenizer_path=path)
        experiment = AutoExperiment.from_experiment_name('supervised', detector=[metric])
        experiment.load_data(target_data)
        print('----------')
        print('Model:', base_model)
        print('Category:', source_subject)
        print('Source DetectLLM:', source_llm)
        print('----------')
        torch.cuda.empty_cache()
        res = experiment.launch(need_finetune=False)
        print('----------')
        print(res[0].train)
        print(res[0].test)
        print('----------')
        if not os.path.exists(llm_result_csv):
            with open(llm_result_csv, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Model', 'Source_Category', 'Source_LLM', 'Target_LLM', 'Mitigate Size', 'Epoch', 'Train F1', 'Test F1'])

        with open(llm_result_csv, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([base_model, source_subject, source_llm, target_llm, 0, 0, round(res[0].train.f1, 4), round(res[0].test.f1, 4)])

        return
    
    else:
        # TODO: use other sizes if necessary
        mitigate_sizes = [100, 200, 500, 800]
        for size in mitigate_sizes:
            mitigate_data = {}
            mitigate_data['train'] = {}
            mitigate_data['train']['text'] = target_data['train']['text'][:size]
            mitigate_data['train']['label'] = target_data['train']['label'][:size]
            mitigate_data['test'] = target_data['test']
        
            print('----------')
            print('Model:', base_model)
            print('Category:', source_subject)
            print('Source DetectLLM:', source_llm)
            print('----------')

            torch.cuda.empty_cache()
            # TODO: run epoch 2 if necessary
            for epoch in [1]:
                config = {
                    'need_finetune': True,
                    'save_path': f"{mitigate_save_dir}/llm/{base_model}/LM-D_{source_subject}_{source_llm}_to_{target_llm}_mitigate_{size}",
                    'epochs': epoch,
                    'batch_size': 32,
                    'disable_tqdm': True
                }
                if size == 0: 
                    config['need_finetune'] = False
                    config['eval'] = True
                    if epoch == 2:
                        break # no need to run twice

                metric = AutoDetector.from_detector_name('LM-D', model_name_or_path=path, tokenizer_path=path)
                # Create an experiment for the given detector
                experiment = AutoExperiment.from_experiment_name('supervised', detector=[metric])

                experiment.load_data(mitigate_data)
                res = experiment.launch(**config)

                # Print results for debugging
                print('----------')
                print(res[0].train)
                print(res[0].test)
                print('----------')
                if not os.path.exists(llm_result_csv):
                    with open(llm_result_csv, 'w', newline='', encoding='utf-8') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(['Model', 'Source_Category', 'Source_LLM', 'Target_LLM', 'Mitigate Size', 'Epoch', 'Train F1', 'Test F1'])

                with open(llm_result_csv, 'a', newline='', encoding='utf-8') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    size = 0 if size == 1 else size
                    csvwriter.writerow([base_model, source_subject, source_llm, target_llm, size, epoch, round(res[0].train.f1, 4), round(res[0].test.f1, 4)])

                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=['domain', 'llm'])
    parser.add_argument('--source_category', type=str, choices=category)
    parser.add_argument('--target_category', type=str, choices=category)
    parser.add_argument('--detectLLM', type=str, choices=llms)
    parser.add_argument('--base_model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--source_llm', type=str, choices=llms)
    parser.add_argument('--target_llm', type=str, choices=llms)

    args = parser.parse_args()
    task = args.task
    source_subject = args.source_category
    target_subject = args.target_category
    detectLLM = args.detectLLM
    base_model = args.base_model
    source_llm = args.source_llm
    target_llm = args.target_llm

    setup_seed(3407)
    if args.task == 'llm':
        assert base_model in ['distilbert-base-uncased', 'roberta-base']
        assert source_subject in category
        assert source_llm in llms
        assert target_llm in llms
        transfer_llm(base_model=base_model,
                    source_subject=source_subject,
                    source_llm=source_llm,
                    target_llm=target_llm
                    )
       
    elif args.task == 'domain':
        assert base_model in ['distilbert-base-uncased', 'roberta-base']
        assert source_subject in category
        assert target_subject in category
        assert detectLLM in llms
        transfer_domain(base_model=base_model,
                        source_subject=source_subject,
                        target_subject=target_subject,
                        detectLLM=detectLLM
                        )
