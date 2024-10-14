# MGTBench 2.0

MGTBench2.0 provides the reference implementations of different machine-generated text (MGT) detection methods.
It is still under continuous development and we will include more detection methods as well as analysis tools in the future.


## Quick Start

### Installation
```
git clone -b release https://github.com/Y-L-LIU/MGTBench-2.0
cd MGTBench-2.0
conda env create -f environment.yml;
conda activate mgtbench2;
```


Check out [`demo.ipynb`](demo.ipynb) for a quick start.
```python
from mgtbench import AutoDetector, AutoExperiment
from mgtbench.loading.dataloader import load

model_name_or_path = '/data1/zzy/gpt2-medium'
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
- Model-based methods:
    - OpenAI Detector [[Ref]](https://arxiv.org/abs/1908.09203);
    - ChatGPT Detector [[Ref]](https://arxiv.org/abs/2301.07597);
    - ConDA [[Ref]](https://arxiv.org/abs/2309.03992) [[Model Weights]](https://www.dropbox.com/s/sgwiucl1x7p7xsx/fair_wmt19_chatgpt_syn_rep_loss1.pt?dl=0);
    - GPTZero [[Ref]](https://gptzero.me/);
    - LM Detector [[Ref]](https://arxiv.org/abs/1911.00650);

## Supported Datasets

- [AITextDetect](https://huggingface.co/AITextDetect)

It contains human written and AI polished text in different categories, including:
- STEM (Physics, Math, Computer, Biology, Chemistry, Electrical, Medicine, Statistics)
- Social Sciences (Education, Management, Economy and Finance)
- Humanities (Art, History, Literature, Philosophy, Law)

From [wiki](https://en.wikipedia.org/wiki/Main_Page), [arxiv](https://arxiv.org/), and [Gutenberg](https://www.gutenberg.org/)



To check the dataset:
```python
from datasets import load_dataset
# ai polished
polish = load_dataset("AITextDetect/AI_Polish_clean",
                      name='gpt35',
                      split='Math',
                      trust_remote_code=True
                    )

# human written
human = load_dataset("AITextDetect/HUMAN-Clean",
                     trust_remote_code=True
                     )
```

## Usage
To run the benchmark on the `AITextDetect` dataset: 
```bash
# specify the model with local path to your model, or model name on huggingface

# distinguish Human vs. Llama3 using LM-D detector
python benchmark.py --detectLLM Llama3 --method LM-D --model /data1/models/distilbert-base-uncased

# distinguish Human vs. gpt3.5 using log-likelihood detector
python benchmark.py --detectLLM gpt35 --method ll --model /data1/zzy/gpt2-medium
```
Note that you can also specify your own datasets on ``dataset_loader.py``.

## Cite
If you find this repo and dataset useful, please consider cite our work
```
@software{rethinkingMGT2024,
  author = {Liu, Yule and Zhong, Zhiyuan and Liao, Yifan and Leng, Jiaqi and Sun, Zhen and Chen, Yang and Gong, Qingyuan and Zhang, Yang and He, Xinlei},
  month = {10},
  title = {{MGTBench-2.0: Rethinking the Machine-Generated Text Detection}},
  url = {https://github.com//Y-L-LIU/MGTBench-2.0},
  version = {2.0.0},
  year = {2024}
}
```
