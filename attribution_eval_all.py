import os
import argparse
import pandas as pd
import subprocess
from multiprocessing import Process, Queue

gpu_pool = [3,5,6,7]  # List of available GPU IDs

def run_eval(gpu_id: int, task_queue: Queue, result_csv: str):
    """Worker function to run evaluation on assigned tasks."""
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()  # Get a task without blocking
        except:
            return

        model_path = task['model_path']
        category = task['category']
        command = f"CUDA_VISIBLE_DEVICES={gpu_pool[gpu_id]} python run_attribution_eval.py " \
                  f"--model_path {model_path} " \
                  f"--category {category} " \
                  f"--output_csv {result_csv}"
        
        print(f"Evaluating on GPU {gpu_pool[gpu_id]}: {category}, model: {model_path}")
        subprocess.run(command, shell=True)

def assign_eval_tasks(gpu_count: int, result_csv: str):
    """Distribute tasks across available GPUs."""
    if not os.path.exists('attribution_results_new.csv'):
        print('No attribution_results_new.csv found, run attribution_train_all.py first')
        exit(1)

    df = pd.read_csv('attribution_results_new.csv')
    max_f1_df = df.loc[df.groupby(['Model', 'Category'])['Test_f1'].idxmax()]
    max_f1_df.reset_index(drop=True, inplace=True)
    max_f1_df.to_csv('attribution_results_best.csv', index=False)

    # Create a queue to store tasks
    task_queue = Queue()

    # Prepare all tasks with the best F1 scores
    for row in max_f1_df.itertuples():
        model_name = row[1]
        category = row[2]
        batch_size = row[3]
        epoch = row[4]

        model_path = f'/data1/zzy/model_attribution/{category}_{model_name}_{batch_size}_{epoch}'
        checkpoints = [c for c in os.listdir(model_path) if c.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
        checkpoint = checkpoints[-1]
        model_path = os.path.join(model_path, checkpoint)

        task_queue.put({
            'model_path': model_path,
            'category': category,
        })

    # Create worker processes for each GPU
    processes = []
    for gpu_id in range(gpu_count):
        p = Process(target=run_eval, args=(gpu_id, task_queue, result_csv))
        p.start()
        processes.append(p)

    # Wait for all worker processes to finish
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_csv', type=str, required=True, help='Path to the output CSV file.')
    args = parser.parse_args()
    
    assign_eval_tasks(
        gpu_count=len(gpu_pool),
        result_csv=args.result_csv
    )