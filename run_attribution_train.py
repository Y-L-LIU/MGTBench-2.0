import argparse
import os
import csv

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_attribution

MODELS = ['Moonshot', 'gpt35', 'Mixtral', 'Llama3']


def train_attribution(model_path, category, epoch, batch_size, save_dir, output_csv):
    data = load_attribution(category)

    # Train model
    model_name_or_path = model_path
    metric = AutoDetector.from_detector_name('LM-D',
                                            model_name_or_path=model_name_or_path,
                                            num_labels=len(MODELS) + 1,
                                            )
    experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric])
    experiment.load_data(data)

    model_name = model_path.split('/')[-1].split('-')[0]
    model_save_dir = f"{save_dir}/{category}_{model_name}_{batch_size}_{epoch}"

    config = {
        'need_finetune': True,
        'save_path': model_save_dir,
        'epochs': epoch,
        'batch_size': batch_size,
        'disable_tqdm': True
        }
    res = experiment.launch(**config)

    print('==========')
    print('category:', category)
    print('train:', res[0].train)
    print('test:', res[0].test)
    
    if os.path.exists(output_csv):
        pass
    else:
        # If CSV doesn't exist, create and write header
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Model', 'Category', 'Batch_Size', 'Epoch', 'Test_f1'])

    # Write results to CSV file
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([model_name, category, batch_size, epoch, round(res[0].test.f1, 4)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--category', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--output_csv', type=str)
    args = parser.parse_args()

    train_attribution(
        model_path=args.model_path,
        category=args.category,
        epoch=args.epoch,
        batch_size=args.batch_size,
        save_dir=args.model_save_dir,
        output_csv=args.output_csv
    )