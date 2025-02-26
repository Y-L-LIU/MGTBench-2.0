import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# llms = ['gpt35', 'Mixtral','Moonshot','Llama3', 'gpt-4omini']
# orders = []
# for k in range(len(llms)):
#     cur = llms.copy()
#     final = cur.pop(k)
#     order = [cur, [final]]
#     orders.append(order)
order = [['gpt35', 'Mixtral','Moonshot', 'Llama3'], ['gpt-4omini']]
orders = [order]
TOPICS = ['STEM', 'Humanities', 'Social_sciences']
from mgtbench import AutoDetector, AutoExperiment
import torch
import numpy as np
import random
import os
import pickle
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(3407)

from mgtbench.loading import load_incremental_topic
result = {}
# Function to load pickle file
# model= '/data1/models/roberta-base'
# model= '/data1/models/distilbert-base-uncased'
# model= '/data1/models/xlm-roberta-base'
model = '/data1/models/deberta-v3-base'

cache_size = [0,100]
for cat in TOPICS:
    result = []
    for size in cache_size:
        for reg in [0,0.2]:
            for order in orders:
                print(f'current order is {order}')
                factor = 4
                print(f'current paras are {reg} and {factor}')
                data = load_incremental_topic(order, cat)
                method = 'incremental'
                #distilbert-base-uncased
                nclass = len(set(data['train'][0]['label']))
                bic = False
                if cache_size==100 and reg==0:
                    bic = True
                metric = AutoDetector.from_detector_name(method, model_name_or_path=model,
                                                        num_labels=nclass, lwf_reg = reg,cache_size=size,bic=bic)
                experiment = AutoExperiment.from_experiment_name('incremental',detector=[metric])
                config = {'need_finetune': True,
                        'need_save': False,
                        'epochs': 2,
                        'lr': 1e-6,
                        'batch_size':64,
                        'save_path': '/data1/lyl/mgtout-1/',
                        'eval':True,
                        'lr_factor': factor
                        }
                # cfg = SupervisedConfig()
                # cfg.update(config)
                experiment.load_data(data)
                res1 = experiment.launch(**config)
                print(res1)
                result.append({'order':order,
                               'lwf_reg':reg,
                               'use_bic':bic,
                                'lr_factor': factor,
                                'cache_size': size,
                                'result': res1})
    basename =os.path.basename(model)
    os.makedirs(f'incremental_{basename}', exist_ok=True)
    with open(f'incremental_{basename}/{cat}_51all.pickle', 'wb') as f:
        pickle.dump(result, f)
