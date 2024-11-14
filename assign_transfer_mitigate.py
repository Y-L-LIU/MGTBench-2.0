import subprocess
from multiprocessing import Process, Queue
import time

CATEGORIES = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
              'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
              'Economy', 'Math', 'Statistics', 'Chemistry']

LLMS = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3']

gpu_pool = [1,2]  # List of available GPUs

def run_transfer_mitigation(gpu_id: int, max_processes: int, queue: Queue):
    active_processes = []
    
    while not queue.empty() or active_processes:
        # Clean up finished processes
        for p in active_processes:
            if not p.is_alive():
                active_processes.remove(p)

        # Start new processes if we have capacity on this GPU
        while len(active_processes) < max_processes and not queue.empty():
            job = queue.get()
            task = job['task']
            if task == 'domain':
                source_category = job['source_category']
                target_category = job['target_category']
                detectLLM = job['detectLLM']
                base_model = job['base_model']
                command = f"CUDA_VISIBLE_DEVICES={gpu_pool[gpu_id]} python transfer_mitigate.py " \
                        f"--task domain " \
                        f"--source_category {source_category} " \
                        f"--target_category {target_category} " \
                        f"--detectLLM {detectLLM} " \
                        f"--base_model {base_model} "
                
                print(f"Running on GPU {gpu_pool[gpu_id]}: Task: {task}, Source: {source_category}, Target: {target_category}, DetectLLM: {detectLLM}, Base Model: {base_model}")
                      
            elif task == 'llm':
                source_category = job['source_category']
                source_llm = job['source_llm']
                target_llm = job['target_llm']
                base_model = job['base_model']
                command = f"CUDA_VISIBLE_DEVICES={gpu_pool[gpu_id]} python transfer_mitigate.py " \
                        f"--task llm " \
                        f"--source_category {source_category} " \
                        f"--source_llm {source_llm} " \
                        f"--target_llm {target_llm} " \
                        f"--base_model {base_model} "
                print(f"Running on GPU {gpu_pool[gpu_id]}: Task: {task}, Source: {source_category}, Source LLM: {source_llm}, Target LLM: {target_llm}, Base Model: {base_model}")

            # Start the subprocess and add it to the active list
            process = Process(target=subprocess.run, args=(command,), kwargs={'shell': True})
            process.start()
            active_processes.append(process)

        # Sleep briefly to avoid busy-waiting
        time.sleep(1)

    # Ensure all processes complete before the function exits
    for p in active_processes:
        p.join()

def assign_run_transfer_mitigation(gpu_count: int, max_processes_per_gpu: int):
    # setup all wanted tasks
    all_tasks = []

    # llm transfer
    for source_category in CATEGORIES:
        for source_llm in LLMS:
            for target_llm in LLMS:
                for base_model in ['distilbert-base-uncased', 'roberta-base']:
                    all_tasks.append({
                        'task': 'llm',
                        'source_category': source_category,
                        'source_llm': source_llm,
                        'target_llm': target_llm,
                        'base_model': base_model
                    })
                    
    # domain transfer
    # for source_category in CATEGORIES:
    for source_category in ['Management']:
        for target_category in CATEGORIES:
            for detectLLM in LLMS:
                for base_model in ['distilbert-base-uncased', 'roberta-base']:
                    all_tasks.append({
                        'task': 'domain',
                        'source_category': source_category,
                        'target_category': target_category,
                        'detectLLM': detectLLM,
                        'base_model': base_model
                    })

    queue = Queue()
    for task in all_tasks:
        queue.put(task)

    processes = []
    # assign tasks to different GPUs with specified max processes per GPU
    for gpu_id in range(gpu_count):
        p = Process(target=run_transfer_mitigation, args=(gpu_id, max_processes_per_gpu, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    print('Remember to setup your available GPUs in gpu_pool')
    print('GPU Pool:', gpu_pool)

    # Example value for max concurrent processes per GPU
    max_processes_per_gpu = 2

    assign_run_transfer_mitigation(
        gpu_count=len(gpu_pool),
        max_processes_per_gpu=max_processes_per_gpu
    )
