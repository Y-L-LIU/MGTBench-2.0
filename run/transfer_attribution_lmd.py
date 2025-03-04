import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import torch
import csv

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_attribution_topic
from mgtbench.utils import setup_seed

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

llms = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']


def get_demo_data(data, train_size, test_size):
    demo = {}
    demo['train'] = {'text': data['train']['text'][:train_size], 'label': data['train']['label'][:train_size]}
    demo['test'] = {'text': data['test']['text'][:test_size], 'label': data['test']['label'][:test_size]}
    return demo


def transfer_domain(model_path, source_topic, target_topic, result_csv):
    setup_seed(3407)
    data_target = load_attribution_topic(topic=target_topic)
    data_target = get_demo_data(data_target, 100, 2000)

    method = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_path, tokenizer_path=model_path)
    experiment = AutoExperiment.from_experiment_name('supervised',detector=[method])
    experiment.load_data(data_target)
    print('----------')
    print('Source topic:', source_topic)
    print('Target topic:', target_topic)
    res = experiment.launch(need_finetune=False,
                            eval=True)
    print('----------')
    print(res[0].train)
    print(res[0].test)
    print('----------')

    if not os.path.exists(result_csv):
        with open(result_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Model', 'Source_topic', 'Target_topic', 'Test F1'])
    else:
        with open(result_csv, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([model_path, source_topic, target_topic, round(res[0].test.f1, 4)])
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_topic', type=str, choices=TOPICS)
    parser.add_argument('--target_topic', type=str, choices=TOPICS)
    parser.add_argument('--model_path', type=str) # trained model path
    parser.add_argument('--result_csv', type=str)

    args = parser.parse_args()
    source_topic = args.source_topic
    target_topic = args.target_topic
    model_path = args.model_path
    result_csv = args.result_csv

    assert source_topic in TOPICS
    assert target_topic in TOPICS
    assert source_topic in model_path
    transfer_domain(model_path=model_path,
                    source_topic=source_topic,
                    target_topic=target_topic,
                    result_csv=result_csv
                    )
