import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import json
import argparse
import torch
import time
import subprocess

from transformers import AutoModelForSequenceClassification
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load
from mgtbench.utils import setup_seed

category = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry']

llms = ['Moonshot']

distilbert = '/data1/models/distilbert-base-uncased'
roberta = '/data1/zzy/roberta-base'
bert = '/data1/zzy/bert-base-uncased'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=category, default="Art")
    parser.add_argument('--target', type=str, choices=category, default="Art")
    parser.add_argument('--detectLLM', type=str, choices=llms, default="Moonshot")
    parser.add_argument('--task', type=str, choices=['old', 'task2','task2_gen', 'task3'],)
    parser.add_argument('--match_data', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    source_subject = args.source
    target_subject = args.target
    detectLLM = args.detectLLM
    task = args.task
    match_data = args.match_data
    eval_all = args.all

    if not os.path.exists(f'transfer'):
        os.makedirs(f'transfer')

    match_tag = '_match' if match_data else ''
    with open(f'{task}_best/best_hyperparams{match_tag}.json', 'r') as f:
        best_hyperparams = json.load(f)

    if not eval_all:
        best_model = best_hyperparams[source_subject][detectLLM]['model']
        model_path = f'/data1/zzy/finetuned/{source_subject}_{detectLLM}_{task}{match_tag}_{best_model}'
        # model_path = '/data1/zzy/finetuned/Biology_Moonshot_task3_roberta'
        # model_path = roberta
        if best_model == 'roberta':
            tokenizer_path = '/data1/zzy/roberta-base'
        else:
            tokenizer_path = '/data1/models/distilbert-base-uncased'
        seed = best_hyperparams[source_subject][detectLLM]['seed']
        best_cut_length = best_hyperparams[source_subject][detectLLM]['cut_length']
        # setup_seed(seed)
        metric1 = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_path, tokenizer_path=tokenizer_path)
        # metric1.model = AutoModelForSequenceClassification.from_pretrained(f'/data1/zzy/finetuned/{datatype}_{detectLLM}_{task}{match_tag}_{args.model}').to('cuda')
        # experiment1 = AutoExperiment.from_experiment_name('supervised',detector=[metric1])
        experiment2 = AutoExperiment.from_experiment_name('supervised',detector=[metric1])
        data_source = load(source_subject, detectLLM, cut_length=best_cut_length, task=task, match=match_data)
        data_target = load(target_subject, detectLLM, cut_length=best_cut_length, task=task, match=match_data)

        # experiment1.load_data(data_source)
        experiment2.load_data(data_target)
        # res = experiment1.launch(need_finetune=False)
        print('----------')
        print('DetectLLM:', detectLLM)
        print('Task:', task)
        print('Match data:', match_data)
        print('Model:', best_model)
        print('----------')
        print('Source Category:', source_subject)
        # print(res[0].train)
        # print(res[0].test.__dict__.__dict__.__dict__)
        print('----------')
        torch.cuda.empty_cache()
        res = experiment2.launch(need_finetune=False)
        print('----------')
        print('Target Category:', target_subject)
        print(res[0].train)
        print(res[0].test)
        print('----------')
        exit()
       
    else:
        # results to be saved
        transfer_results = {}
        transfer_results = {}
        for source_cat in category:
            transfer_results[source_cat] = []

        for source_cat in category[7:]:
            for tranfer_cat in category[:]:
                for llmname in llms:
                    best_model = best_hyperparams[source_cat][detectLLM]['model']
                    model_path = f'/data1/zzy/finetuned/{source_cat}_{detectLLM}_{task}{match_tag}_{best_model}'
                    # model_path = roberta
                    if best_model == 'roberta':
                        tokenizer_path = '/data1/zzy/roberta-base'
                    else:
                        tokenizer_path = '/data1/models/distilbert-base-uncased'
                    seed = best_hyperparams[source_cat][detectLLM]['seed']
                    best_cut_length = best_hyperparams[source_cat][detectLLM]['cut_length']

                    # if source_cat == tranfer_cat:
                    #     transfer_results[source_cat][tranfer_cat] =  {'train':res[0].train.__dict__, 'test': res[0].test.__dict__.__dict__.__dict__}
                    #     continue
                    # print(model_path)
                    setup_seed(seed)
                    metric1 = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_path, tokenizer_path=tokenizer_path)
                    # metric1.model = AutoModelForSequenceClassification.from_pretrained(f'/data1/zzy/finetuned/{datatype}_{detectLLM}_{task}{match_tag}_{args.model}').to('cuda')
                    # experiment1 = AutoExperiment.from_experiment_name('supervised',detector=[metric1])
                    experiment2 = AutoExperiment.from_experiment_name('supervised',detector=[metric1])
                    # data_source = load(source_cat, detectLLM, cut_length=best_cut_length, task=task, match=match_data)
                    data_target = load(tranfer_cat, detectLLM, cut_length=best_cut_length, task=task, match=match_data)
                    # experiment1.load_data(data_source)
                    experiment2.load_data(data_target)
                    # res1 = experiment1.launch(need_finetune=False)
                    print('----------')
                    print('DetectLLM:', detectLLM)
                    print('Task:', task)
                    print('Match data:', match_data)
                    print('Model:', best_model)
                    print('----------')
                    print('Source Category:', source_cat)
                    # print(res[0].train)
                    # print(res[0].test.__dict__.__dict__.__dict__)
                    print('Target Category:', tranfer_cat)
                    # print('----------')
                    torch.cuda.empty_cache()
                    res = experiment2.launch(need_finetune=False)
                    print('----------')
                    print(res[0].train)
                    print(res[0].test)
                    print('----------')
                    transfer_results[source_cat].append({tranfer_cat : {'train':res[0].train.__dict__, 'test': res[0].test.__dict__}})

            with open(f'tranfer/{detectLLM}_{task}{match_tag}_transfer_results2.json', 'w') as f:
                json.dump(transfer_results, f)

        with open(f'tranfer/{detectLLM}_{task}{match_tag}_transfer_results2.json', 'w') as f:
            json.dump(transfer_results, f)


# split running into two parts, change:
# 1. category[7:] or category[:7]
# 2. result1.json or result2.json

                


        
