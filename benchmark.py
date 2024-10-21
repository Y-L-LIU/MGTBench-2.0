import torch
import json
import argparse
import os
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed
setup_seed(3407)
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
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--save', type=lambda x: (str(x).lower() == 'true'), default='true')
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='/data1/lyl/mgtout/')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    args = parser.parse_args()

    datasource = args.dataset
    config['epochs'] = args.epochs
    config['need_save'] = args.save
    config['batch_size'] = args.batch_size
    config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    config['lr'] = args.lr
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
        data['train']['text'] = data['train']['text']
        data['train']['label'] = data['train']['label']
        data['test']['text'] = data['test']['text']
        data['test']['label'] = data['test']['label']
        experiment.load_data(data)
        model1 = model.rstrip('/').split('/')[-1]
        config['name'] = f"{method}_{args.detectLLM}_{cat}"
        config['save_path'] = os.path.join(args.save_path,model1,config['name'])
        res = experiment.launch(**config)
        print(f'===== {cat} - {args.detectLLM} - {model}=====')
        print('train:', res[0].train)
        print('test:', res[0].test)
        print()

        results[cat] = res[0].test.__dict__
    
    # save the results
    model = model.replace('/', '_')
    with open(f'result/{args.method}_{model}_{args.detectLLM}_{args.lr}.json', 'w') as f:
        json.dump(results, f)
