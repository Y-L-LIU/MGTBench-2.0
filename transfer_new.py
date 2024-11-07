import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import json
import argparse
import torch

from transformers import AutoModelForSequenceClassification
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed

category = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry']

llms = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3']

distilbert = '/data1/models/distilbert-base-uncased'
# roberta = '/data1/zzy/roberta-base'
# bert = '/data1/zzy/bert-base-uncased'
import re
base_dir = '/data1/lyl/mgtout'  # Update this to your base folder path

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


# Function to run the new transfer experiment
def transfer_across_detectLLM(model_path, source_LLM, cat):
    # Results dictionary to store all transfer results
    transfer_results = {
        'LLM': source_LLM,
        'model': model_path,
        'source': {},
        'target': []
    }
    detectLLMs = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3']
    # Iterate over each detectLLM to perform the transfer task
    for detectLLM in detectLLMs:
        # Get the path for the specific model and detectLLM
        path = get_path(model_path, source_LLM, cat)
        metric = AutoDetector.from_detector_name('LM-D', model_name_or_path=path, tokenizer_path=path)

        # Create an experiment for the given detector
        experiment = AutoExperiment.from_experiment_name('supervised', detector=[metric])

        # Load the data for the fixed source category
        data = load('AITextDetect', detectLLM=detectLLM, category=cat)
        experiment.load_data(data)
        
        print('----------')
        print('Model:', model_path)
        print('Category:', cat)
        print('Source DetectLLM:', detectLLM)
        print('----------')

        # Clear CUDA cache if using GPU
        torch.cuda.empty_cache()

        # Launch the experiment without fine-tuning
        res = experiment.launch(need_finetune=False)

        # Print results for debugging
        print('----------')
        print(res[0].train)
        print(res[0].test)
        print('----------')
        if detectLLM == source_LLM:
            transfer_results['source'] = {
            'detectLLM': detectLLM,
            'train': res[0].train.__dict__,
            'test': res[0].test.__dict__
        }
        else:
            transfer_results['target'].append({
                'detectLLM': detectLLM,
                'train': res[0].train.__dict__,
                'test': res[0].test.__dict__
            })

        # Intermediate saving of results
        tmp_file = f'tmp/{model_path}_{source_LLM}_{cat}_transfer.json'
        with open(tmp_file, 'w') as f:
            json.dump(transfer_results, f)

    # Save final results
    final_file = f'transfer_result_1/{model_path}_{source_LLM}_{cat}_transfer.json'
    with open(final_file, 'w') as f:
        json.dump(transfer_results, f)
# Run the transfer experiment with fixed category


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=category, default="Art")
    parser.add_argument('--target', type=str, choices=category, default="Art")
    parser.add_argument('--detectLLM', type=str, choices=llms, default="Moonshot")
    parser.add_argument('--model', type=str, default='distilbert-base-uncased', )
    parser.add_argument("--task", type=str, choices=['domain', 'llm'])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    source_subject = args.source
    target_subject = args.target
    detectLLM = args.detectLLM
    model_path = args.model
    eval_all = args.all
    setup_seed(3407)
    if not eval_all:
        # setup_seed(seed)
        path1 = get_path(model_path, detectLLM, source_subject)
        metric1 = AutoDetector.from_detector_name('LM-D', model_name_or_path=path1, tokenizer_path=path1)
        path2 = get_path(model_path, detectLLM, target_subject)
        metric2 = AutoDetector.from_detector_name('LM-D', model_name_or_path=path2, tokenizer_path=path2)
        experiment1 = AutoExperiment.from_experiment_name('supervised',detector=[metric1])
        experiment2 = AutoExperiment.from_experiment_name('supervised',detector=[metric2])
        data_source = load('AITextDetect', detectLLM=detectLLM, category=source_subject)
        data_target = load('AITextDetect', detectLLM=detectLLM, category=target_subject)

        experiment1.load_data(data_source)
        experiment2.load_data(data_target)
        res1 = experiment1.launch(need_finetune=False)
        print('----------')
        print('DetectLLM:', detectLLM)
        print('Source Category:', source_subject)
        print(res1[0].train)
        print(res1[0].test)
        print('----------')
        torch.cuda.empty_cache()
        res2 = experiment2.launch(need_finetune=False)
        print('----------')
        print('Target Category:', target_subject)
        print(res2[0].train)
        print(res2[0].test)
        print('----------')
        log = {}
        log['detectLLM'] = detectLLM
        log['model'] = model_path
        source_res = {'category': source_subject, 'train': res1[0].train.__dict__, 'test': res1[0].test.__dict__}
        target_res = {'category': target_subject, 'train': res2[0].train.__dict__, 'test': res2[0].test.__dict__}
        log['source'] = source_res
        log['target'] = target_res
        json.dump(log, open(f'{detectLLM}_{source_subject}_{target_subject}_transfer.json', 'w'))
        exit()
       
    elif args.task == 'domain':
        # results to be saved
        transfer_results = {}
        transfer_results['detectLLM'] = detectLLM
        transfer_results['model'] = model_path
        transfer_results['source'] = {}
        transfer_results['target'] = []

        # from source to target, given one detectLLM
        for tranfer_cat in category[:]:
            # setup_seed(seed)
            path2 = get_path(model_path, detectLLM, source_subject)
            metric2 = AutoDetector.from_detector_name('LM-D', model_name_or_path=path2, tokenizer_path=path2)

            # metric1 = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_path, tokenizer_path=model_path)
            experiment2 = AutoExperiment.from_experiment_name('supervised',detector=[metric2])

            data_target = load('AITextDetect', detectLLM=detectLLM, category=tranfer_cat)
            experiment2.load_data(data_target)
            print('----------')
            print('DetectLLM:', detectLLM)
            print('Source Category:', source_subject)
            print('----------')
            print('Target Category:', tranfer_cat)
            torch.cuda.empty_cache()
            res = experiment2.launch(need_finetune=False)
            print('----------')
            print(res[0].train)
            print(res[0].test)
            print('----------')
            if source_subject == tranfer_cat:
                transfer_results['source'] = {'category': source_subject, 'train': res[0].train.__dict__, 'test': res[0].test.__dict__}
            else:
                transfer_results['target'].append({'category': tranfer_cat, 'train': res[0].train.__dict__, 'test': res[0].test.__dict__})

            # for intermediate saving
            json.dump(transfer_results, open(f'tmp/{detectLLM}_{source_subject}_transfer.json', 'w'))

        json.dump(transfer_results, open(f'tmp/{model_path}_{detectLLM}_{source_subject}_transfer.json', 'w'))
    elif args.task == 'llm':
        for cat in category:
            transfer_across_detectLLM(model_path, detectLLM,cat)
