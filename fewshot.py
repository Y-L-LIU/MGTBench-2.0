import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# llms = ['gpt35', 'Mixtral','Moonshot','Llama3', 'gpt-4omini']
# orders = []
# for k in range(len(llms)):
#     if k<2:
#         continue
#     cur = llms.copy()
#     final = cur.pop(k)
#     order = [cur, [final]]
#     orders.append(order)
order = [['gpt35', 'Mixtral','Moonshot',], ['Llama3','gpt-4omini']]
orders = [order]
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
# TOPICS = ['STEM', ]
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.experiment import SupervisedConfig
import torch
import numpy as np
import random
import os
import pickle
from mgtbench.methods import supervised
from mgtbench.loading import load_incremental_topic
result = {}
# Function to load pickle file
models= ['/data1/models/roberta-base', '/data1/models/distilbert-base-uncased', '/data1/models/deberta-v3-base']
# models= ['/data1/models/roberta-base']
# '/data1/models/roberta-base',
methods = ['baseline', 'rn', 'generate']
kshot = [1,5,20]
for cat in TOPICS:
    for model in models:
        result = []
        for order in orders:
            print(f'current order is {order}')
            factor = 4
            print(f'current model: {model}')
            data = load_incremental_topic(order, cat)
            #distilbert-base-uncased
            nclass = len(set(data['train'][0]['label']))
            config = {'need_finetune': True,
                        'need_save': False,
                        'epochs': 2,
                        'lr': 1e-6,
                        'batch_size':64,
                        'save_path': '/data1/lyl/mgtout-1/',
                        'eval':True,
                        'lr_factor': factor,
                        'classifier':'MLP'
                        }
            cfg = SupervisedConfig()
            cfg.update(config)
            supervise = AutoDetector.from_detector_name('LM-D', model_name_or_path=model,
                                                        num_labels=nclass,)
            supervise.finetune(data['train'][0],cfg)
            for method in methods:
                metric = AutoDetector.from_detector_name(method, model=supervise.model,tokenizer=supervise.tokenizer,
                                                        num_labels=nclass,kshot=1)
                experiment = AutoExperiment.from_experiment_name('fewshot',detector=[metric])
                config['need_finetune'] = False
                experiment.load_data(data)
                experiment.launch(**config)
                for shot in kshot:
                    trials = []
                    for _ in range(5):
                        res1 = experiment.launch(**config)
                        trials.append(res1)
                    result.append({'order':order,
                                'kshot':shot,
                                'cat':cat,
                                'model': model,
                                'method':method,
                                'result': trials})
                basename =os.path.basename(model)
                os.makedirs(f'fewshot_{basename}_32all', exist_ok=True)
                with open(f'fewshot_{basename}_32all/{cat}.pickle', 'wb') as f:
                    pickle.dump(result, f)
