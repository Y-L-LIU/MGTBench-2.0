import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load_attribution


def eval_attribution(model_path, category, output_csv):
    data = load_attribution(category)

    # Load model
    model_name_or_path = model_path
    metric = AutoDetector.from_detector_name('LM-D',
                                            model_name_or_path=model_name_or_path,
                                            num_labels=5
                                            )
    experiment = AutoExperiment.from_experiment_name('supervised',detector=[metric])
    experiment.load_data(data)

    config = {
        'need_finetune': False,
        'disable_tqdm': True,
        'eval': True
        }
    # add eval mode, do not predict on train set
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
            csvwriter.writerow(['Model', 'Category', 'Test_f1'])

    # Write results to CSV file
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([model_path, category, round(res[0].test.f1, 4)])

    # draw confusion matrix
    arr = res[0].test.conf_m
    save_dir = 'figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_name = 'roberta' if 'roberta' in model_path else 'distilbert'
    save_path = os.path.join(save_dir, f'{model_name}_{category}_confusion_matrix.png')
    plot_confusion_matrix(arr, category, save_path)


label_to_class = {0: 'Human', 1: 'Moonshot', 2: 'gpt35', 3: 'Mixtral', 4: 'Llama3'}
class_names = ['Human', 'Moonshot', 'gpt35', 'Mixtral', 'Llama3']


def plot_confusion_matrix(conf_m, category, save_path):
    # Calculate the percentage for each cell in the confusion matrix
    conf_m_percent = conf_m.astype('float') / conf_m.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_m_percent, annot=True, fmt='.2f',  # Format as percentage
                 cmap='Blues',
                 cbar=False, 
                 xticklabels=class_names,
                 yticklabels=class_names
                 )
    model_name = save_path.split('/')[-1].split('_')[0]
    plt.xlabel('Predicted LLMs')
    plt.ylabel('True LLMs')
    plt.title(f'{category}_{model_name}')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--category', type=str)
    parser.add_argument('--output_csv', type=str)
    args = parser.parse_args()

    eval_attribution(
        model_path=args.model_path,
        category=args.category,
        output_csv=args.output_csv
    )