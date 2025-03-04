import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import torch
import csv

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_attribution_topic
from mgtbench.utils import setup_seed
from mgtbench.auto import DetectOutput, Metric

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

llms = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']

TRAIN_SIZE = 1000
TEST_SIZE = 2000
# change to your own path
MITIGATE_SAVE_DIR = '/data_sda/zhiyuan/transfer_mitigate_attribution_topics'


def get_demo_data(data, train_size, test_size):
    demo = {}
    demo['train'] = {'text': data['train']['text'][:train_size], 'label': data['train']['label'][:train_size]}
    demo['test'] = {'text': data['test']['text'][:test_size], 'label': data['test']['label'][:test_size]}
    return demo

def log_domain_result(csv_file: str, 
                      train_result: DetectOutput,
                      test_result: DetectOutput,
                      source_topic: str, 
                      target_topic: str, 
                      base_model: str,
                      mitigate_size: int,
                      epoch: int,
                      batch_size: int
                      ):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Base_model', 'Source_topic', 'Target_topic',
                                'Mitigate_size', 'Epoch', 'Batch_size',
                                'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc',
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([base_model, source_topic, target_topic, 
                            mitigate_size, epoch, batch_size,
                            round(train_result.acc, 4), round(train_result.precision, 4), round(train_result.recall, 4), round(train_result.f1, 4), round(train_result.auc, 4),
                            round(test_result.acc, 4), round(test_result.precision, 4), round(test_result.recall, 4), round(test_result.f1, 4), round(test_result.auc, 4)])
        

def transfer_domain_mitigate(model_path, source_topic, target_topic, result_csv):
    setup_seed(3407)
    data_target = load_attribution_topic(topic=target_topic)

    if 'distil' in model_path:
        base_model = 'distilbert'
    elif 'roberta' in model_path:
        base_model = 'roberta'
    else:
        raise ValueError('Unknown model')
    
    if source_topic == target_topic:
        detector = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_path, tokenizer_path=model_path)
        experiment = AutoExperiment.from_experiment_name('supervised',detector=[detector])
        # test only, cuz the same topic
        exp_data = get_demo_data(data_target, 10, TEST_SIZE)
        experiment.load_data(exp_data)
        print('----------')
        print('Source Category:', source_topic)
        print('Target Category:', target_topic)
        res = experiment.launch(need_finetune=False)
        print('----------')
        print(res[0].test)
        print('----------')
        log_domain_result(result_csv, res[0].train, res[0].test, source_topic, target_topic, base_model, 0, None, None)
        return

    # TODO: change size if needed
    data_sizes  = [0, 100, 200, 300, 500, 800]
    print('----------')
    print('Source Category:', source_topic)
    print('Target Category:', target_topic)    
    print('----------')
    for size in data_sizes:
        mitigate_data = get_demo_data(data_target, size, TEST_SIZE)
        print('Mitigate Size:', size)
        torch.cuda.empty_cache()

        # TODO: run epoch 2 if necessary
        for epoch in [1]:
            for batch_size in [32]:
                model_save_dir = f"{MITIGATE_SAVE_DIR}/domain/{base_model}/LM-D_{source_topic}_to_{target_topic}_mitigate_{size}_epoch_{epoch}_batch_{batch_size}"
                config = {
                    'need_finetune': True,
                    'save_path': model_save_dir,
                    'epochs': epoch,
                    'batch_size': batch_size,
                    'disable_tqdm': True
                }
                if size == 0:
                    config['need_finetune'] = False
                    config['eval'] = True

                detector = AutoDetector.from_detector_name('LM-D', model_name_or_path=model_path, tokenizer_path=model_path)
                experiment = AutoExperiment.from_experiment_name('supervised',detector=[detector])
                experiment.load_data(mitigate_data)
                res = experiment.launch(**config)
                print('----------')
                print(res[0].test)
                if res[0].train is None:
                    res[0].train = Metric(0, 0, 0, 0, 0)
                log_domain_result(result_csv, res[0].train, res[0].test, source_topic, target_topic, base_model, size, epoch, batch_size)
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_topic', type=str, choices=TOPICS)
    parser.add_argument('--target_topic', type=str, choices=TOPICS)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--result_csv', type=str)

    args = parser.parse_args()
    source_topic = args.source_topic
    target_topic = args.target_topic
    model_path = args.model_path
    result_csv = args.result_csv

    assert source_topic in TOPICS
    assert target_topic in TOPICS
    assert source_topic in model_path
    transfer_domain_mitigate(model_path=model_path,
                    source_topic=source_topic,
                    target_topic=target_topic,
                    result_csv=result_csv
                    )
