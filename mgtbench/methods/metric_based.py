import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Union
from ..utils import timeit, get_clf_results, assert_tokenizer_consistency
from .IntrinsicDim import PHD
from torch.utils.data import DataLoader

from ..auto import BaseDetector
from ..loading import load_pretrained
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from sklearn.metrics import accuracy_score, f1_score, roc_curve
import warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# # Under development
# def get_phd(text, base_model, base_tokenizer, DEVICE):
#     # default setting
#     MIN_SUBSAMPLE = 40
#     INTERMEDIATE_POINTS = 7
#     alpha=1.0
#     solver = PHD(alpha=alpha, metric='euclidean', n_points=9)

#     text = text[:200]
#     inputs = base_tokenizer(text, truncation=True, max_length=1024, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         outp = base_model(**inputs)

#     # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
#     mx_points = inputs['input_ids'].shape[1] - 2
#     mn_points = MIN_SUBSAMPLE
#     step = ( mx_points - mn_points ) // INTERMEDIATE_POINTS

#     t1 = time.time()
#     res = solver.fit_transform(outp[0][0].cpu().numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step, point_jump=step)
#     print(time.time() - t1, "Seconds")
#     return res

class MetricBasedDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        self.model = kargs.get('model', None)
        self.tokenizer = kargs.get('tokenizer', None)
        self.device = kargs.get('device', None)
        self.classifier = LogisticRegression()
        if not self.model or not  self.tokenizer:
            model_name_or_path = kargs.get('model_name_or_path', None)
            if not model_name_or_path :
                raise ValueError('You should pass the model_name_or_path or a model instance, but none is given')
            quantitize_bit = kargs.get('load_in_k_bit', None)
            self.model, self.tokenizer = load_pretrained(model_name_or_path, quantitize_bit)
        if not isinstance(self.model, PreTrainedModel) or not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise ValueError('Expect PreTrainedModel, PreTrainedTokenizer, got', type(self.model), type(self.tokenizer))
        

class LLDetector(MetricBasedDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name, **kargs)


    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=1024,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                labels = tokenized.input_ids
                result.append( -self.model(**tokenized, labels=labels).loss.item())
        return result if isinstance(text, list) else result[0]
    
    def find_threshold(self, train_scores, train_labels):
        # Sort scores to get possible threshold values
        print("Finding best threshold for f1...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_accuracy = 0
        for t in thresholds:
            predictions = (train_scores > t).astype(int)
            accuracy = accuracy_score(train_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t
        self.threshold = best_threshold
        return best_threshold, best_accuracy
 
class RankDetector(MetricBasedDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name, **kargs)

    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=1024,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                logits = self.model(**tokenized).logits[:, :-1]
                labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True)
                    == labels.unsqueeze(-1)).nonzero()
            assert matches.shape[
                1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float() + 1  # convert to 1-indexed rank
            log = kargs.get("log", False)
            if log:
                ranks = torch.log(ranks)
            result.append(ranks.float().mean().item())
        return result if isinstance(text, list) else result[0]

    def find_threshold(self, train_scores, train_labels):
        # Sort scores to get possible threshold values
        print("Finding best threshold for f1...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_accuracy = 0
        for t in thresholds:
            predictions = (train_scores < t).astype(int)
            accuracy = accuracy_score(train_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t
        self.threshold = best_threshold
        return best_threshold, best_accuracy


class LRRDetector(RankDetector, LLDetector):
    def __init__(self, name, **kargs) -> None:
        RankDetector.__init__(self, name, **kargs)
        LLDetector.__init__(self, name, model=self.model, tokenizer = self.tokenizer)

    def detect(self, text):
        p_rank_origin = np.array(RankDetector.detect(self, text, log=True))
        p_ll_origin = np.array(LLDetector.detect(self, text))
        epsilon = 1e-10
        return p_ll_origin / (p_rank_origin + epsilon)
        # return p_ll_origin/p_rank_originw

    def find_threshold(self, train_scores, train_labels):
        # Sort scores to get possible threshold values
        print("Finding best threshold for f1...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_accuracy = 0
        for t in thresholds:
            predictions = (train_scores < t).astype(int)
            accuracy = accuracy_score(train_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t
        self.threshold = best_threshold
        return best_threshold, best_accuracy


class RankGLTRDetector(MetricBasedDetector):
    def __init__(self,name, **kargs) -> None:
        super().__init__(name,**kargs)

    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=1024,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                logits = self.model(**tokenized).logits[:, :-1]
                labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True)
                    == labels.unsqueeze(-1)).nonzero()
            assert matches.shape[
                1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float()
            res = np.array([0.0, 0.0, 0.0, 0.0])
            for i in range(len(ranks)):
                if ranks[i] < 10:
                    res[0] += 1
                elif ranks[i] < 100:
                    res[1] += 1
                elif ranks[i] < 1000:
                    res[2] += 1
                else:
                    res[3] += 1
            if res.sum() > 0:
                res = res / res.sum()
            result.append(res)
            
        return result if isinstance(text, list) else result[0]


class EntropyDetector(MetricBasedDetector):
    def __init__(self,name, **kargs) -> None:
        super().__init__(name,**kargs)

    def detect(self, text, **kargs):
        result = []
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt").to(self.model.device)
                logits = self.model(**tokenized).logits[:, :-1]
                neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
                result.append( -neg_entropy.sum(-1).mean().item())
        return result if isinstance(text, list) else result[0]

    def find_threshold(self, train_scores, train_labels):
        # Sort scores to get possible threshold values
        print("Finding best threshold for f1...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_accuracy = 0
        for t in thresholds:
            predictions = (train_scores < t).astype(int)
            accuracy = accuracy_score(train_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t
        self.threshold = best_threshold
        return best_threshold, best_accuracy


class BinocularsDetector(BaseDetector):
    def __init__(self,name, **kargs) -> None:
        self.name = 'Binoculars'
        observer_name_or_path = kargs.get('observer_model_name_or_path', "tiiuae/falcon-7b")
        performer_name_or_path = kargs.get('performer_model_name_or_path', "tiiuae/falcon-7b-instruct")
        print(observer_name_or_path, performer_name_or_path)
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.DEVICE_1 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.DEVICE_2 = torch.device("cuda:1") if torch.cuda.device_count() > 1 else self.DEVICE_1

        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map=self.DEVICE_1,
                                                                   torch_dtype=torch.bfloat16
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map=self.DEVICE_2,
                                                                    torch_dtype=torch.bfloat16
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()
        self.classifier = LogisticRegression()

        # selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
        self.BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
        self.BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]
        mode = kargs.get('mode', "accuracy")
        self.mode = mode
        if mode == "low-fpr":
            self.threshold = self.BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = self.BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.threshold_strategy = kargs.get('threshold', 'default')
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        max_length = kargs.get('max_length', 512)
        self.max_token_observed = max_length
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    def perplexity(self, encoding: BatchEncoding,
                logits: torch.Tensor,
                median: bool = False,
                temperature: float = 1.0):
        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = encoding.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

        if median:
            ce_nan = (self.ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                    masked_fill(~shifted_attention_mask.bool(), float("nan")))
            ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

        else:
            ppl = (self.ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
                shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
            ppl = ppl.to("cpu").float().numpy()

        return ppl

    def entropy(self,
                p_logits: torch.Tensor,
                q_logits: torch.Tensor,
                encoding: BatchEncoding,
                pad_token_id: int,
                median: bool = False,
                sample_p: bool = False,
                temperature: float = 1.0):
        vocab_size = p_logits.shape[-1]
        total_tokens_available = q_logits.shape[-2]
        p_scores, q_scores = p_logits / temperature, q_logits / temperature

        p_proba = self.softmax_fn(p_scores).view(-1, vocab_size)

        if sample_p:
            p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

        # q_score and p_score must be the same shape
        q_scores = q_scores.view(-1, vocab_size)

        ce = self.ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
        padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

        if median:
            ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
            agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
        else:
            agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

        return agg_ce

    def _tokenize(self, batch: list[str]) -> BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings
    
    @torch.inference_mode()
    def _get_logits(self, encodings: BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(self.DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(self.DEVICE_2)).logits
        if self.DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        # should have matched tokenizer!
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = self.perplexity(encodings, performer_logits)
        x_ppl = self.entropy(observer_logits.to(self.DEVICE_1), performer_logits.to(self.DEVICE_1),
                        encodings.to(self.DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores
    
    def find_threshold(self, train_scores, train_labels):
        # Sort scores to get possible threshold values
        print(f"Finding best threshold for {self.mode}...")
        if self.threshold_strategy == 'default':
            assert self.threshold == self.BINOCULARS_ACCURACY_THRESHOLD or self.threshold == self.BINOCULARS_FPR_THRESHOLD
            return self.threshold
        
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_accuracy = 0
        if self.mode == "low-fpr":
            scores = train_scores
            fpr, tpr, roc_thresholds = roc_curve(train_labels, scores)
            # Find the threshold where FPR is closest to 0.01%
            target_fpr = 0.0001
            idx = np.where(fpr <= target_fpr)[0][-1]  # Closest index where FPR <= 0.01%
            self.threshold = roc_thresholds[idx]

        elif self.mode == "accuracy":
            for t in thresholds:
                predictions = (train_scores < t).astype(int)
                accuracy = f1_score(train_labels, predictions)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = t

            self.threshold = best_threshold
        return best_threshold

    def change_mode(self, mode):
        if mode not in ["low-fpr", "accuracy"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
    
    def detect(self, text, **kargs):
        predictions = []
        for idx in tqdm(range(len(text)), desc="Detecting"):
            binoculars_scores = np.array(self.compute_score(text[idx]))
            predictions.append(binoculars_scores)

        return predictions