# MGTBench 2.0:Rethinking the Machine-Generated Text Detection

Official Repo for [On the Generalization and Adaptation Ability of Machine-Generated Text Detectors in Academic Writing](https://arxiv.org/pdf/2412.17242)
![overview](https://github.com/user-attachments/assets/d8a0d4f7-79ad-4425-a3ee-65b0132e591b)


## Quick Start

### Installation
```bash
git clone https://github.com/Y-L-LIU/MGTBench-2.0
cd MGTBench-2.0
conda env create -f mgtbench2.yml;
conda activate mgtbench2;
# you may need mirror for faster installation
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
pip install -r requirements.txt
```


Check out [`notebook/detection.ipynb`](notebook/detection.ipynb) for a quick start.
```python
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load

model_name_or_path = 'openai-community/gpt2-medium'
metric = AutoDetector.from_detector_name('ll', 
                                            model_name_or_path=model_name_or_path)
experiment = AutoExperiment.from_experiment_name('threshold',detector=[metric])

data_name = 'AITextDetect'
detectLLM = 'gpt35'
category = 'Art'
data = load(data_name, detectLLM, category)
experiment.load_data(data)
res = experiment.launch()

print('train:', res[0].train)
print('test:', res[0].test)
```

### Dataloader

An exmaple usage is provided in [`check_dataloader.ipynb`](notebook/check_dataloader.ipynb).

For our dataset, we support multiple ways to load:

* load by category:
```python
'''
supported LLMs and detect categories:

categories = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 'Economy', 'Math', 'Statistics', 'Chemistry']

llms = ['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini']

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

'''
data_name = 'AITextDetect'
detectLLM = 'Llama3'
category = 'Art'
data = load(data_name, detectLLM, category) #2 classes 
data = load_attribution(data_name, detectLLM) #all classes
```

* load by topics (recommended):
```python
from mgtbench.loading.dataloader import load_topic_data, load_attribution_topic
data = load_topic_data(detectLLM, topic)
# Humanities Social_sciences
data = load_attribution_topic('Social_sciences')
```

Additionally, we support loading the data in an incremental way:

```python
# two stages. first stage includes 5 classes (with human) and the second stage incude 1 classes
order = [['gpt35', 'Mixtral','Moonshot','Llama3',],['gpt-4omini']]
# fives stages. first stage includes 2 classes (with human) and each remaining stage includes 1 class
order = [['Moonshot'],['Mixtral'],['gpt35'],['Llama3'],['gpt-4omini']]
from mgtbench.loading import load_incremental_topic, load_incremental
data = load_incremental_topic(order, "Social_sciences")
```

## Supported Methods
Currently, we support the following methods (continuous updating):
- Metric-based methods:
    - Log-Likelihood [[Ref]](https://arxiv.org/abs/1908.09203);
    - Rank [[Ref]](https://arxiv.org/abs/1906.04043);
    - Log-Rank [[Ref]](https://arxiv.org/abs/2301.11305);
    - Entropy [[Ref]](https://arxiv.org/abs/1906.04043);
    - GLTR Test 2 Features (Rank Counting) [[Ref]](https://arxiv.org/abs/1906.04043);
    - DetectGPT [[Ref]](https://arxiv.org/abs/2301.11305);
    - LRR [[Ref]](https://arxiv.org/abs/2306.05540);
    - NPR [[Ref]](https://arxiv.org/abs/2306.05540);
    - DNA-GPT [[Ref]](https://arxiv.org/abs/2305.17359);
    - Fast-DetectGPT [[Ref]](https://arxiv.org/abs/2310.05130);
    - Binoculars [[Ref]](https://arxiv.org/abs/2401.12070);
- Model-based methods:
    - OpenAI Detector [[Ref]](https://arxiv.org/abs/1908.09203);
    - ChatGPT Detector [[Ref]](https://arxiv.org/abs/2301.07597);
    - ConDA [[Ref]](https://arxiv.org/abs/2309.03992) [[Model Weights]](https://www.dropbox.com/s/sgwiucl1x7p7xsx/fair_wmt19_chatgpt_syn_rep_loss1.pt?dl=0);
    - GPTZero [[Ref]](https://gptzero.me/);
    - RADAR [[Ref]](https://arxiv.org/abs/2307.03838);
    - LM Detector [[Ref]](https://arxiv.org/abs/1911.00650);

## Supported Datasets

- [AITextDetect](https://huggingface.co/datasets/AITextDetect/AI_Polish_clean)

It contains human written and AI polished text in different categories, including:
- STEM (Physics, Math, Computer, Biology, Chemistry, Electrical, Medicine, Statistics)
- Social Sciences (Education, Management, Economy and Finance)
- Humanities (Art, History, Literature, Philosophy, Law)

From [wiki](https://en.wikipedia.org/wiki/Main_Page), [arxiv](https://arxiv.org/), and [Gutenberg](https://www.gutenberg.org/)


## Usage

Checkout the [run](run) folder for more examples.

Note that you can also specify your own datasets in ``dataloader.py``.

## Cite
If you find this repo and dataset useful, please consider cite our work
```
@misc{liu2025generalizationadaptationabilitymachinegenerated,
      title={On the Generalization and Adaptation Ability of Machine-Generated Text Detectors in Academic Writing}, 
      author={Yule Liu and Zhiyuan Zhong and Yifan Liao and Zhen Sun and Jingyi Zheng and Jiaheng Wei and Qingyuan Gong and Fenghua Tong and Yang Chen and Yang Zhang and Xinlei He},
      year={2025},
      eprint={2412.17242},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.17242}, 
}

@inproceedings{he2024mgtbench,
author = {He, Xinlei and Shen, Xinyue and Chen, Zeyuan and Backes, Michael and Zhang, Yang},
title = {{Mgtbench: Benchmarking machine-generated text detection}},
booktitle = {{ACM SIGSAC Conference on Computer and Communications Security (CCS)}},
pages = {},
publisher = {ACM},
year = {2024}
}
```
