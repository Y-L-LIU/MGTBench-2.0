import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..auto import BaseDetector
from torch.utils.data import DataLoader
from tqdm import tqdm

class RadarDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        local_path = '/data_sda/zhiyuan/models/Radar'
        if os.path.exists(local_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(local_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        else:
            self.model  = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
            self.tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
        self.model.eval()
        self.model.to('cuda')
        print('Radar Detector is loaded')

    def detect(self, text, **kargs):
        disable_tqdm = kargs.get('disable_tqdm', False)
        result = []
        if not isinstance(text, list):
            text = [text]
        n_positions = self.model.config.max_position_embeddings
        if n_positions<1000:
            n_positions=512
        else:
            n_positions=4096
            
        num_labels = self.model.config.num_labels
        assert num_labels == 2, "Radar only supports binary classification"
        pos_bit=0
        for batch in tqdm(DataLoader(text), disable=disable_tqdm):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=512,
                    return_tensors="pt",
                    padding=True,
                    truncation = True
                ).to(self.model.device)
                result.append(self.model(**tokenized).logits.softmax(-1)[:, pos_bit].item())
       

        return result if isinstance(text, list) else result[0]