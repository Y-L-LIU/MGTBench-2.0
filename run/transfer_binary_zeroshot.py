import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import torch
import csv

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_topic_data
from mgtbench.utils import setup_seed
from mgtbench.auto import DetectOutput

METHODS = ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR',
           'Binoculars', 'fast-detectGPT']

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

LLMS = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']

# change to model path
gpt_neo = '/data_sda/zhiyuan/models/gpt-neo-2.7B'
gpt_j = '/data_sda/zhiyuan/models/gpt-j-6B'
falcon_7b = '/data_sda/zhiyuan/models/falcon-7b'
falcon_7b_instruct = '/data_sda/zhiyuan/models/falcon-7b-instruct'
llama2_chat = '/data1/models/Llama-2-7b-chat-hf'


def log_result(csv_file: str, 
               result: DetectOutput, 
               source_topic: str, 
               target_topic: str, 
               source_llm: str,
               target_llm: str,
               method: str, 
               base_model='None'):
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['method', 'classifier', 'source_topic', 'target_topic', 'source_llm', 'target_llm', 'base_model', 
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    if method in ['Binoculars', 'fast-detectGPT']:
        base_model = None

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        classifier = result.name if result.name else 'None'
        csvwriter.writerow([method, classifier, source_topic, target_topic, source_llm, target_llm, base_model,
                            round(result.test.acc, 4), round(result.test.precision, 4), round(result.test.recall, 4), round(result.test.f1, 4), round(result.test.auc, 4)])


def get_demo_data(data, train_size, test_size):
    demo = {}
    demo['train'] = {'text': data['train']['text'][:train_size], 'label': data['train']['label'][:train_size]}
    demo['test'] = {'text': data['test']['text'][:test_size], 'label': data['test']['label'][:test_size]}
    return demo


def transfer_domain_and_llm(method, result_csv):
    setup_seed(3407)

    assert method in METHODS, f"Method {method} not zero-shot!"

    # domain transfer
    for detectLLM in LLMS:
        for source_topic in TOPICS:
            # pretrained classifiers
            config = {
                # change to saved path
                'logistic_path': f'/data_sda/zhiyuan/zeroshot_single_model/logistic_{method}_{source_topic}_{detectLLM}.joblib',
                'threshold_path': f'/data_sda/zhiyuan/zeroshot_single_model/threshold_{method}_{source_topic}_{detectLLM}.json',
            }
            if method == 'Binoculars':
                observer_model_name_or_path = falcon_7b
                performer_model_name_or_path = falcon_7b_instruct

                detector = AutoDetector.from_detector_name('Binoculars', 
                                                            observer_model_name_or_path=observer_model_name_or_path,
                                                            performer_model_name_or_path= performer_model_name_or_path,
                                                            max_length=1024,
                                                            mode='accuracy', # accuracy (f1)
                                                            threshold='new',
                                                            pretrained_logistic_path = config['logistic_path'],
                                                            pretrained_threshold_path = config['threshold_path'],
                                                            )
                experiment = AutoExperiment.from_experiment_name('threshold',detector=[detector])

            elif method in ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR']:
                detector = AutoDetector.from_detector_name(method, 
                                                            model_name_or_path=llama2_chat,
                                                            pretrained_logistic_path = config['logistic_path'],
                                                            pretrained_threshold_path = config['threshold_path'],
                                                            )
                experiment = AutoExperiment.from_experiment_name('threshold',detector=[detector])
            
            elif method in ['fast-detectGPT']:
                scoring_model_name_or_path = gpt_neo
                reference_model_name_or_path = gpt_j
                detector = AutoDetector.from_detector_name('fast-detectGPT', 
                                                            scoring_model_name_or_path=scoring_model_name_or_path,
                                                            reference_model_name_or_path=reference_model_name_or_path,
                                                            pretrained_logistic_path = config['logistic_path'],
                                                            pretrained_threshold_path = config['threshold_path'],
                                                            )
                experiment = AutoExperiment.from_experiment_name('perturb', detector=[detector])

        # domain transfer
        for target_topic in TOPICS:
            data_target = load_topic_data(detectLLM=detectLLM, topic=target_topic)
            data_target = get_demo_data(data_target, 10, 100)
            experiment.load_data(data_target)
            print('----------')
            print('Source Category:', source_topic)
            print('Target Category:', target_topic)
            print('Method:', method)
            print('DetectLLM:', detectLLM)
            exp_config = {
                'use_pretrained': True,
            }
            res = experiment.launch(**exp_config)
            print('----------')

            source_llm = detectLLM
            target_llm = detectLLM
            log_result(csv_file=result_csv, 
                        result=res[0], 
                        source_topic=source_topic, 
                        target_topic=target_topic,
                        source_llm=source_llm,
                        target_llm=target_llm,
                        method=method, 
                        base_model=llama2_chat
                        )
            if len(res) > 1:
                log_result(csv_file=result_csv,
                            result=res[1],
                            source_topic=source_topic,
                            target_topic=target_topic,
                            source_llm=source_llm,
                            target_llm=target_llm,
                            method=method,
                            base_model=llama2_chat
                            )
                
        # LLM transfer
        for target_llm in LLMS:
            data_target = load_topic_data(detectLLM=target_llm, topic=source_topic)
            data_target = get_demo_data(data_target, 10, 100)
            experiment.load_data(data_target)
            print('----------')
            print('Target Category:', target_topic)
            print('Method:', method)
            print('DetectLLM:', target_llm)
            exp_config = {
                'use_pretrained': True,
            }
            res = experiment.launch(**exp_config)
            print('----------')
            source_llm = detectLLM

            log_result(csv_file=result_csv, 
                        result=res[0], 
                        source_topic=source_topic, 
                        target_topic=target_topic,
                        source_llm=source_llm,
                        target_llm=target_llm, 
                        method=method, 
                        base_model=llama2_chat
                        )
            if len(res) > 1:
                log_result(csv_file=result_csv,
                            result=res[1],
                            source_topic=source_topic,
                            target_topic=target_topic,
                            source_llm=source_llm,
                            target_llm=target_llm,
                            method=method,
                            base_model=llama2_chat
                            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--result_csv', type=str)

    args = parser.parse_args()
    method = args.method
    result_csv = args.result_csv

    transfer_domain_and_llm(method=method,
                    result_csv=result_csv
                    )
