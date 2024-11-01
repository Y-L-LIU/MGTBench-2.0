import argparse
import subprocess
from multiprocessing import Process, Queue

CATEGORIES = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
              'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
              'Economy', 'Math', 'Statistics', 'Chemistry']

# model_save_dir = '/data1/zzy/model_attribution'
model_save_dir = '/data_sda/zhiyuan/model_attribution'
# model_save_dir = '/data1/zzy/temp'

output_csv = 'attribution_results_new.csv'
# output_csv = 'temp.csv'

gpu_pool = [0,5]

def run_training(gpu_id: int, queue: Queue, model_save_dir: str, output_csv: str):
    while not queue.empty():
        task = queue.get()
        category = task['category']
        batch_size = task['batch_size']
        epoch = task['epoch']
        base_model = task['base_model']
        command = f"CUDA_VISIBLE_DEVICES={gpu_pool[gpu_id]} python run_attribution_train.py " \
                  f"--category {category} " \
                  f"--batch_size {batch_size} " \
                  f"--epoch {epoch} " \
                  f"--model_path {base_model} " \
                  f"--model_save_dir {model_save_dir} " \
                  f"--output_csv {output_csv} "
        
        print(f"Training on GPU {gpu_pool[gpu_id]}: {category}, Batch size: {batch_size}, Epoch: {epoch}, Base model: {base_model}")
        subprocess.run(command, shell=True)

def assign_run_attribution(gpu_count: int, model_save_dir: str, output_csv: str):
    # setup all wanted tasks
    all_categories = CATEGORIES
    batch_size = [32, 64]
    epochs = [2, 3]
    base_model = ['/data1/models/distilbert-base-uncased', '/data1/zzy/roberta-base']

    all_tasks = []
    for category in all_categories:
        for bs in batch_size:
            for epoch in epochs:
                for bm in base_model:
                    all_tasks.append({
                        'category': category,
                        'batch_size': bs,
                        'epoch': epoch,
                        'base_model': bm
                    })

    queue = Queue()
    for task in all_tasks:
        queue.put(task)

    processes = []
    # assign tasks to different GPUs
    for gpu_id in range(gpu_count):
        p = Process(target=run_training, args=(gpu_id, queue, model_save_dir, output_csv))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--output_csv', type=str) # training results
    args = parser.parse_args()

    print('Remember to setup your available GPUs in gpu_pool')
    print('GPU Pool:', gpu_pool)

    model_save_dir = args.model_save_dir
    output_csv = args.output_csv

    assign_run_attribution(
        gpu_count=len(gpu_pool),
        model_save_dir=model_save_dir,
        output_csv=output_csv
    )