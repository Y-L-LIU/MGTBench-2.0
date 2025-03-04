import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
order = [['gpt35', 'Mixtral','Moonshot','Llama3',], ['gpt-4omini']]
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


def get_detector(name):
    if name == 'fast-detectGPT':
        scoring_model_name_or_path = '/data_sda/zhiyuan/models/gpt-neo-2.7B'
        reference_model_name_or_path = '/data_sda/zhiyuan/models/gpt-j-6B'
        detector = AutoDetector.from_detector_name('fast-detectGPT', 
                                                    scoring_model_name_or_path=scoring_model_name_or_path,
                                                    reference_model_name_or_path= reference_model_name_or_path
                                                    )
    elif name == 'll':
        model_name_or_path = '/data1/models/Llama-2-7b-chat-hf'
        detector = AutoDetector.from_detector_name('ll',
                                                model_name_or_path=model_name_or_path)
    elif name == 'rank_GLTR':
        model_name_or_path= '/data1/models/Llama-2-7b-chat-hf'
        detector = AutoDetector.from_detector_name('rank_GLTR',
                                                model_name_or_path=model_name_or_path)
    elif name == 'Binoculars':
        observer_model_name_or_path = '/data_sda/zhiyuan/models/falcon-7b'
        performer_model_name_or_path = '/data_sda/zhiyuan/models/falcon-7b-instruct'
        detector = AutoDetector.from_detector_name('Binoculars', 
                                                    observer_model_name_or_path=observer_model_name_or_path,
                                                    performer_model_name_or_path= performer_model_name_or_path,
                                                    max_length=1024,
                                                    mode='low-fpr', # accuracy (f1) or low-fpr
                                                    # 'default' or 'new', default is the threshold used in the paper, 'new' is the threshold calculated on the new training set
                                                    threshold='new' 
                                                    )
    return detector

from mgtbench.loading import load_incremental_topic
import os
import pickle
from multiprocessing import Process, Queue

# Function to run an experiment
def run_experiment_sequence(cat, name, cache_sizes, order, device, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)  # Assign specific CUDA device
    results = []
    for size in cache_sizes:
        data = load_incremental_topic(order, cat)
        metric = get_detector(name)
        experiment = AutoExperiment.from_experiment_name('incremental_threshold', detector=[metric])
        experiment.load_data(data)
        res1 = experiment.launch()
        results.append({'name': name, 'cache_size': size, 'result': res1, 'category': cat})
    queue.put(results)
# Parameters
cache_sizes = [0]
cuda_devices = [5,6,7,3]  # List of available CUDA devices

# Running experiments concurrently
for cat in TOPICS:
    processes = []
    queue = Queue()
    device_index = 0
    results = []

    # Parallelize over `name`
    for name in ['fast-detectGPT','ll', 'rank_GLTR','Binoculars']:
        # Get the next CUDA device
        device = cuda_devices[device_index % len(cuda_devices)]
        device_index += 1

        # Create and start a process
        p = Process(target=run_experiment_sequence, args=(cat, name, cache_sizes, order, device, queue))
        processes.append(p)
        p.start()

    # Collect results from all processes
    for p in processes:
        p.join()  # Ensure all processes are finished

    while not queue.empty():
        results.extend(queue.get())

    # Save results
    with open(f'incremental_metric2/{cat}_result.pickle', 'wb') as f:
        pickle.dump(results, f)