import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import torch
import csv

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_attribution_topic
from mgtbench.utils import setup_seed
from mgtbench.auto import DetectOutput

METHODS = ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR',
           'Binoculars', 'fast-detectGPT']

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

TEST_SIZE = 2000

# change the path to your own path
gpt_neo = '/data_sda/zhiyuan/models/gpt-neo-2.7B'
gpt_j = '/data_sda/zhiyuan/models/gpt-j-6B'
falcon_7b = '/data_sda/zhiyuan/models/falcon-7b'
falcon_7b_instruct = '/data_sda/zhiyuan/models/falcon-7b-instruct'
llama2_chat = '/data1/models/Llama-2-7b-chat-hf'

result_csv = './results/transfer_attribution.csv'

def log_result(csv_file: str, result: DetectOutput, source_topic: str, target_topic: str, method: str, base_model='None'):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['method', 'classifier', 'source_topic', 'target_topic', 'base_model', 'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    if method in ['Binoculars', 'fast-detectGPT']:
        base_model = None

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        classifier = result.name if result.name else 'None'
        csvwriter.writerow([method, classifier, source_topic, target_topic, base_model,
                             round(result.test.acc, 4), round(result.test.precision, 4), round(result.test.recall, 4), round(result.test.f1, 4), round(result.test.auc, 4)])

def get_demo_data(data, train_size, test_size):
    demo = {}
    demo['train'] = {'text': data['train']['text'][:train_size], 'label': data['train']['label'][:train_size]}
    demo['test'] = {'text': data['test']['text'][:test_size], 'label': data['test']['label'][:test_size]}
    return demo


def get_method_scores(method):
    setup_seed(3407)
    assert method in METHODS, f"Method {method} not zero-shot!"

    # check if there is pre-saved scores
    unsaved = False
    for topic in TOPICS:
        exp_config = {
            'save_test_score': True,
            # change to your own path
            'test_score_x_path': f'/data_sda/zhiyuan/zeroshot_scores/{method}_{topic}_attribution_x_test_{TEST_SIZE}.npy',
            'test_score_y_path': f'/data_sda/zhiyuan/zeroshot_scores/{method}_{topic}_attribution_y_test_{TEST_SIZE}.npy',
        }
        if os.path.exists(exp_config['test_score_x_path']) and os.path.exists(exp_config['test_score_y_path']):
            print('Already saved!')
        else:
            unsaved = True
            break

    if not unsaved:
        print('All saved!')
        return

    if method == 'Binoculars':
        observer_model_name_or_path = falcon_7b
        performer_model_name_or_path = falcon_7b_instruct

        detector = AutoDetector.from_detector_name('Binoculars', 
                                                    observer_model_name_or_path=observer_model_name_or_path,
                                                    performer_model_name_or_path= performer_model_name_or_path,
                                                    max_length=1024,
                                                    mode='accuracy', # accuracy (f1)
                                                    threshold='new',
                                                    )
        experiment = AutoExperiment.from_experiment_name('threshold',detector=[detector])

    elif method in ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR']:
        detector = AutoDetector.from_detector_name(method, 
                                                    model_name_or_path=llama2_chat,
                                                    )
        experiment = AutoExperiment.from_experiment_name('threshold',detector=[detector])
    
    elif method in ['fast-detectGPT']:
        scoring_model_name_or_path = gpt_neo
        reference_model_name_or_path = gpt_j
        detector = AutoDetector.from_detector_name('fast-detectGPT', 
                                                    scoring_model_name_or_path=scoring_model_name_or_path,
                                                    reference_model_name_or_path=reference_model_name_or_path,
                                                    )
        experiment = AutoExperiment.from_experiment_name('perturb', detector=[detector])

    # predict scores for all the data

    for topic in TOPICS:
        data = load_attribution_topic(topic=topic)
        data = get_demo_data(data, 10, test_size=TEST_SIZE)
        experiment.load_data(data)
        print('----attribution------')
        print('Topic:', topic)
        print('Method:', method)
        exp_config = {
            'attribution': True,
            'save_test_score': True,
            # change to your own path
            'test_score_x_path': f'/data_sda/zhiyuan/zeroshot_scores/{method}_{topic}_attribution_x_test_{TEST_SIZE}.npy',
            'test_score_y_path': f'/data_sda/zhiyuan/zeroshot_scores/{method}_{topic}_attribution_y_test_{TEST_SIZE}.npy',
        }
        if os.path.exists(exp_config['test_score_x_path']) and os.path.exists(exp_config['test_score_y_path']):
            print('Already saved!')
        else:
            res = experiment.launch(**exp_config)
            print('----------')
            assert os.path.exists(exp_config['test_score_x_path']), f"Test score x not saved!"
            assert os.path.exists(exp_config['test_score_y_path']), f"Test score y not saved!"


def transfer_domain(method, result_csv):
    setup_seed(3407)

    assert method in METHODS, f"Method {method} not zero-shot!"

    if method == 'Binoculars':
        observer_model_name_or_path = falcon_7b
        performer_model_name_or_path = falcon_7b_instruct

        detector = AutoDetector.from_detector_name('Binoculars', 
                                                    observer_model_name_or_path=observer_model_name_or_path,
                                                    performer_model_name_or_path= performer_model_name_or_path,
                                                    max_length=1024,
                                                    mode='accuracy', # accuracy (f1)
                                                    threshold='new'
                                                    )
        experiment = AutoExperiment.from_experiment_name('threshold',detector=[detector])

    elif method in ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR']:
        detector = AutoDetector.from_detector_name(method, 
                                                    model_name_or_path=llama2_chat
                                                    )
        experiment = AutoExperiment.from_experiment_name('threshold',detector=[detector])
    
    elif method in ['fast-detectGPT']:
        scoring_model_name_or_path = gpt_neo
        reference_model_name_or_path = gpt_j
        detector = AutoDetector.from_detector_name('fast-detectGPT', 
                                                    scoring_model_name_or_path=scoring_model_name_or_path,
                                                    reference_model_name_or_path=reference_model_name_or_path,
                                                    )
        experiment = AutoExperiment.from_experiment_name('perturb', detector=[detector])


    for source_topic in TOPICS:
        # pretrained classifiers
        config = {
            # change to your own path
            'pretrained_logistic_path': f'/data_sda/zhiyuan/zeroshot_attribution_model/logistic_{method}_{source_topic}_attribution.joblib',
            'pretrained_svm_path': f'/data_sda/zhiyuan/zeroshot_attribution_model/svm_{method}_{source_topic}_attribution.joblib',
            'pretrained_threshold_path': None
        }
        detector.load_pretrained_classifier(**config)

        for target_topic in TOPICS:
            data_target = load_attribution_topic(topic=target_topic)
            data_target = get_demo_data(data_target, 10, test_size=TEST_SIZE)
            experiment.load_data(data_target)
            print('----------')
            print('Source Category:', source_topic)
            print('Target Category:', target_topic)
            exp_config = {
                'attribution': True,
                'use_pretrained': True,
                # saved scores, change to your own path
                'use_saved_score': True,
                'test_score_x_path': f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_attribution_x_test_{TEST_SIZE}.npy',
                'test_score_y_path': f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_attribution_y_test_{TEST_SIZE}.npy',
            }
            res = experiment.launch(**exp_config)
            print('----------')
            log_result(csv_file=result_csv, 
                        result=res[0], 
                        source_topic=source_topic, 
                        target_topic=target_topic, 
                        method=method, 
                        base_model=llama2_chat
                        )
            if len(res) > 1:
                log_result(csv_file=result_csv,
                            result=res[1],
                            source_topic=source_topic,
                            target_topic=target_topic,
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

    get_method_scores(method=method)
    transfer_domain(method=method,
                    result_csv=result_csv
                    )
