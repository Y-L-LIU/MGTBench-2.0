import torch
import json
import argparse

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load

config = {'need_finetune': True,
          'need_save': False,
          'epochs': 1
        }

categories = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry'
            ]

METHOD_MAPPING = {
    'll': 'threshold',
    'rank': 'threshold',
    'rank_GLTR': 'threshold',
    'entropy': 'threshold',
    'detectGPT': 'perturb',
    'NPR': 'threshold',
    'LRR': 'threshold',
    'LM-D': 'supervised',
    'demasq': 'demasq'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detectLLM', type=str, default='Moonshot', choices=['Moonshot', 'gpt35', 'Mixtral', 'Llama3'])
    parser.add_argument('--method', type=str, default='ll', choices=['gptzero', 'll', 'rank', 'rank_GLTR', 'entropy', 'detectGPT', 'NPR', 'LRR', 'LM-D', 'demasq'])
    parser.add_argument('--model', type=str, default='distilbert')
    parser.add_argument('--dataset', type=str, default="AITextDetect")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save', type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()

    datasource = args.dataset
    config['epochs'] = args.epochs
    model = args.model
    save = args.save
    method = args.method

    results = {} # benmark over different categories
    results['dataset'] = datasource
    results['model'] = model
    results['method'] = args.method
    results['detectLLM'] = args.detectLLM
    for cat in categories:
        results[cat] = {}

    # run all the experiments
    cnt = 0
    for cat in categories:
        torch.cuda.empty_cache()
        metric = AutoDetector.from_detector_name(method, model_name_or_path=model)
        experiment = AutoExperiment.from_experiment_name(METHOD_MAPPING[method],detector=[metric])

        data = load(name=datasource, detectLLM=args.detectLLM, category=cat)
        data['train']['text'] = data['train']['text'][:100]
        data['train']['label'] = data['train']['label'][:100]
        data['test']['text'] = data['test']['text'][:100]
        data['test']['label'] = data['test']['label'][:100]
        experiment.load_data(data)

        res = experiment.launch(**config)
        print(f'===== {cat} =====')
        print('train:', res[0].train)
        print('test:', res[0].test)
        print()

        results[cat] = res[0].test.__dict__
    
    # save the results
    model = model.replace('/', '_')
    with open(f'{datasource}_{args.method}_{model}_{args.detectLLM}.json', 'w') as f:
        json.dump(results, f)
