import argparse
import os
import csv

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_attribution_topic
from mgtbench.auto import DetectOutput

# add more models here
MODELS = ['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini']

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

METHODS = ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR',
           'Binoculars',
           'fast-detectGPT',
        #    'detectGPT', 'NPR', 'DNA-GPT',
            ]

# change to your local model path
gpt_neo = '/data_sda/zhiyuan/models/gpt-neo-2.7B'
gpt_j = '/data_sda/zhiyuan/models/gpt-j-6B'
falcon_7b = '/data_sda/zhiyuan/models/falcon-7b'
falcon_7b_instruct = '/data_sda/zhiyuan/models/falcon-7b-instruct'
llama2_chat = '/data1/models/Llama-2-7b-chat-hf'


def log_result(csv_file: str, result: DetectOutput, method: str, classifier: str, topic: str, base_model='None'):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['method', 'classifier', 'topic', 'base_model', 'train_f1', 'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    if method in ['Binoculars', 'fast-detectGPT']:
        base_model = None

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([method, classifier, topic, base_model, round(result.train.f1, 4),
                             round(result.test.acc, 4), round(result.test.precision, 4), round(result.test.recall, 4), round(result.test.f1, 4), round(result.test.auc, 4)])


def get_demo_data(data, train_size, test_size):
    demo = {}
    demo['train'] = {'text': data['train']['text'][:train_size], 'label': data['train']['label'][:train_size]}
    demo['test'] = {'text': data['test']['text'][:test_size], 'label': data['test']['label'][:test_size]}
    return demo


def train_attribution(model_path, method, output_csv):
    assert method in METHODS, f"Method {method} not zero-shot!"

    if method in ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR', 'Binoculars']:
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
        else:
            detector = AutoDetector.from_detector_name(method, model_name_or_path=model_path)

        experiment = AutoExperiment.from_experiment_name('threshold',detector=[detector])
    
    elif method in ['fast-detectGPT']:
        scoring_model_name_or_path = gpt_neo
        reference_model_name_or_path = gpt_j
        detector = AutoDetector.from_detector_name('fast-detectGPT', 
                                                    scoring_model_name_or_path=scoring_model_name_or_path,
                                                    reference_model_name_or_path= reference_model_name_or_path
                                                    )
        experiment = AutoExperiment.from_experiment_name('perturb', detector=[detector])
            

    for topic in TOPICS:
        data = load_attribution_topic(topic)
        data = get_demo_data(data, 1000, 2000)

        experiment.load_data(data)
        config = {
            'attribution': True,
            # change to your local save path
            'logistic_path': f'/data_sda/zhiyuan/zeroshot_model/logistic_{method}_{topic}_attribution.joblib',
            'svm_path': f'/data_sda/zhiyuan/zeroshot_model/svm_{method}_{topic}_attribution.joblib',
        }
        res = experiment.launch(**config)

        print('==========')
        print('topic:', topic)
        print('train:', res[0].train)
        print('test:', res[0].test)
        log_result(output_csv, res[0], method, 'logistic', topic, model_path)
        if len(res) > 1:
            print('==========')
            print('topic:', topic)
            print('train:', res[1].train)
            print('test:', res[1].test)
            log_result(output_csv, res[1], method, 'svm', topic, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--output_csv', type=str)
    args = parser.parse_args()

    train_attribution(
        model_path=args.model_path,
        method=args.method,
        output_csv=args.output_csv
    )