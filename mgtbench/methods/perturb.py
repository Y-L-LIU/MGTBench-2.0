from ..auto import BaseDetector
from ..methods import LLDetector, RankDetector
from ..loading import load_pretrained,load_pretrained_mask
import numpy as np
from sklearn.metrics import f1_score, roc_curve
import re
import torch
import random
import time
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def load_mask_model(args, mask_model):
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    # base_model.cpu()
    if not args.random_fills:
        mask_model.to(args.DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, buffer_size, pct, ceil_pct=False):
    tokens = text.split(' ')

    # Note that you can also comment these line out if you have enough memory
    if len(tokens) > 1024:
        tokens = tokens[:1024]
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM
    # increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")])
            for text in texts]


# replace each masked span with a sample from T5 self.mask_model
def replace_masks(texts, mask_model, mask_tokenizer, mask_top_p, DEVICE):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt",
                            padding=True).to(DEVICE)
    outputs = mask_model.generate(
        **tokens,
        max_length=150,
        do_sample=True,
        top_p=mask_top_p,
        num_return_sequences=1,
        eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(
            zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(
        args,
        texts,
        mask_model,
        mask_tokenizer,
        tokenizer,
        ceil_pct=False):
    span_length = args.span_length
    buffer_size = args.buffer_size
    mask_top_p = args.mask_top_p
    pct = args.pct_words_masked
    DEVICE = args.DEVICE
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(
            x, span_length, buffer_size, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(
            masked_texts, mask_model, mask_tokenizer, mask_top_p, DEVICE)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right
        # number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            masked_texts = [
                tokenize_and_mask(
                    x,
                    span_length,
                    pct,
                    ceil_pct) for idx,
                x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(
                masked_texts, mask_model, mask_tokenizer, mask_top_p, DEVICE)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(
                masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize tokenizer
            tokens = tokenizer(
                texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * \
                (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(
                tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(
                0, tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them
            # with random non-special tokens
            while any(tokenizer.decode(
                    x) in tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(
                    0, tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = tokenizer.batch_decode(
                tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(
                x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from
            # FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(
                        f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[
                    0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts



class PerturbBasedDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        self.name = name
        model_name_or_path = kargs.get('model_name_or_path', None)
        mask_model_name_or_path = kargs.get('mask_model_name_or_path', None)
        if not model_name_or_path or not mask_model_name_or_path :
            raise ValueError('You should pass the model_name_or_path and mask_model_name_or_path, but ',model_name_or_path,mask_model_name_or_path, 'are given')
        # scoring/source model
        self.model, self.tokenizer = load_pretrained(model_name_or_path)
        # mask filling model
        self.mask_model, self.mask_tokenizer = load_pretrained_mask(mask_model_name_or_path)
        self.ceil_pct = kargs.get('ceil_pct', False)

    def perturb_once(self, texts, perturb_config, chunk_size=20):
        outputs = []
        for i in tqdm(range(0,len(texts),chunk_size)):
            outputs.extend(perturb_texts_(perturb_config,
                                        texts[i:i + chunk_size],
                                        self.mask_model,
                                        self.mask_tokenizer,
                                        self.tokenizer,
                                        ceil_pct=self.ceil_pct))
        return outputs

    def perturb(self, text, label,  n_perturbations, perturb_config):
        p_text = self.perturb_once([x for x in text for _ in range(n_perturbations)], perturb_config)
        
        for _ in range(perturb_config.n_perturbation_rounds - 1):
            try:
                p_text = self.perturb_once(p_text, perturb_config)
            except AssertionError:
                break

        assert len(p_text) == len(text) * \
            n_perturbations, f"Expected {len(text) * n_perturbations} perturbed samples, got {len(p_text)}"
        data = {'text':[],
                'label':[],
                'perturbed_text':[]}

        for idx in range(len(text)):
                data["text"].append(text[idx])
                data["label"].append( label[idx])
                data["perturbed_text"].extend(p_text[idx * n_perturbations: (idx + 1) * n_perturbations])
        data['len'] =  len(text)
        return data


class DetectGPTDetector(PerturbBasedDetector, LLDetector):
    def __init__(self, name, **kargs) -> None:
        PerturbBasedDetector.__init__(self, name, **kargs)
        LLDetector.__init__(self,name,model=self.model, tokenizer = self.tokenizer)
        self.threshold = None

    def find_threshold(self, train_scores, train_labels):
        print(f"Finding best threshold for f1 score...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_f1 = 0
        for threshold in thresholds:
            # machine's score is largeer, human's score is smaller
            predictions = train_scores > threshold
            f1 = f1_score(train_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold
        return best_threshold

    def detect(self, text, label, config):
        perturb_config = config
        print('Running perturb on the given texts')
        data = self.perturb(text, label,perturb_config.n_perturbations,perturb_config)
        print('Perturb finished.')
        p_ll_origin = LLDetector.detect(self, data["text"])
        p_ll_origin = np.array(p_ll_origin)
        p_ll = LLDetector.detect(self,data["perturbed_text"])
        perturbed_ll_mean = []
        perturbed_ll_std = []
        for batch in DataLoader(p_ll, batch_size=perturb_config.n_perturbations):
            batch = batch.numpy()
            perturbed_ll_mean.append(np.mean(batch))
            perturbed_ll_std.append(np.std(batch) if len(batch)>1 else 1)
        assert len(p_ll_origin) == len(perturbed_ll_mean)
        if perturb_config.criterion_score == 'd':
            predictions = p_ll_origin - perturbed_ll_mean
        elif perturb_config.criterion_score == 'z':
            perturbed_ll_std = [std if std > 0 else 1 for std in perturbed_ll_std]
            predictions = (p_ll_origin - perturbed_ll_mean)/perturbed_ll_std
        return predictions


# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
class FastDetectGPTDetector():
    def __init__(self, name, **kargs) -> None:
        self.name = name
        scoring_model_name_or_path = kargs.get('scoring_model_name_or_path', None)
        reference_model_name_or_path = kargs.get('reference_model_name_or_path', None)
        if not scoring_model_name_or_path:
            raise ValueError('You should pass the scoring_model_name_or_path, but ',scoring_model_name_or_path, 'are given')
        # scoring model
        self.scoring_model, self.scoring_tokenizer = load_pretrained(scoring_model_name_or_path)
        self.scoring_model.eval()
        if reference_model_name_or_path:
            if reference_model_name_or_path != scoring_model_name_or_path:
                self.reference_model, self.reference_tokenizer = load_pretrained(reference_model_name_or_path)
                self.reference_model.eval()
        if kargs.get('discrepancy_analytic', False):
            self.criterion_fn = 'discrepancy_analytic'
        else:
            self.criterion_fn = 'discrepancy'

    def get_samples(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = 10000
        lprobs = torch.log_softmax(logits, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])
        return samples

    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood.mean(dim=1)

    # Copyright (c) Guangsheng Bao.
    def get_sampling_discrepancy(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self.get_samples(logits_ref, labels)
        log_likelihood_x = self.get_likelihood(logits_score, labels)
        log_likelihood_x_tilde = self.get_likelihood(logits_score, samples)
        miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
        discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
        return discrepancy.item()

    def get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1) # get log likelihood of the correct token as in the original sequence (next token prediction)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1) # enumerate over vocabulary
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref) # enumerate over vocabulary
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt() # mean over sequence
        discrepancy = discrepancy.mean()
        return discrepancy.item()
    
    def find_threshold(self, train_scores, train_labels):
        print(f"Finding best threshold for f1 score...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_f1 = 0
        for threshold in thresholds:
            # machine's score is largeer, human's score is smaller
            predictions = train_scores > threshold
            f1 = f1_score(train_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold
        return best_threshold
    
    def detect(self, text, label, config):
        seed = config.seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        predictions = []
        if self.criterion_fn == 'discrepency':
            criterion_fn = self.get_sampling_discrepancy
        else:
            criterion_fn = self.get_sampling_discrepancy_analytic

        for idx in tqdm(range(len(text)), desc="Detecting"):
            tokenized = self.scoring_tokenizer(text[idx],
                                               return_tensors="pt", 
                                               padding=True, 
                                               return_token_type_ids=False,
                                               truncation=True,  # Truncate to max model length
                                               ).to(self.scoring_model.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                if not hasattr(self, 'reference_model'): # reference model == scoring model
                    logits_ref = logits_score
                else:
                    tokenized = self.reference_tokenizer(text[idx],
                                                         return_tensors="pt",
                                                         padding=True, 
                                                         return_token_type_ids=False,
                                                         truncation=True
                                                         ).to(self.reference_model.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = self.reference_model(**tokenized).logits[:, :-1]
                crit = criterion_fn(logits_ref, logits_score, labels)
                predictions.append(crit)

        return predictions        


class NPRDetector(PerturbBasedDetector, RankDetector):
    def __init__(self, name, **kargs) -> None:
        PerturbBasedDetector.__init__(self,name, **kargs)
        RankDetector.__init__(self,name,model=self.model, tokenizer = self.tokenizer)

    def detect(self, text, label, config):
        perturb_config = config
        print('Running perturb on the given texts')
        data = self.perturb(text, label, perturb_config.n_perturbations,perturb_config)
        print('Perturb finished.')

        p_rank_origin = RankDetector.detect(self, data["text"], log=True)
        p_rank_origin = np.array(p_rank_origin)

        p_rank = RankDetector.detect(self,data["perturbed_text"], log=True)
        perturbed_rank_mean = []
        for batch in DataLoader(p_rank, batch_size=perturb_config.n_perturbations):
            batch = batch.numpy()
            perturbed_rank_mean.append(np.mean(batch))
        # print(len(p_rank_origin), len(perturbed_rank_mean))
        assert len(p_rank_origin) == len(perturbed_rank_mean)
        predictions = perturbed_rank_mean/p_rank_origin

        return predictions


# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
class DNAGPTDetector(BaseDetector):
    def __init__(self,name, **kargs) -> None:
        self.name = 'DNA-GPT'
        self.sampler = self.PrefixSampler(**kargs)
        self.mode = kargs.get('mode', 'accuracy')

    class PrefixSampler:
        def __init__(self, **kargs):
            self.base_model_name_or_path = kargs.get('base_model_name_or_path', 'gpt2')
            self.batch_size = kargs.get('batch_size', 5)
            self.regen_number = kargs.get('regen_number', 5)
            self.temperature = kargs.get('temperature', 1.0)
            self.truncate_ratio = kargs.get('truncate_ratio', 0.5)
            # self.top_p = kargs.get('top_p', 0.96)
            # self.top_k = kargs.get('top_k', 40)

            self.base_model, self.base_tokenizer = load_pretrained(model_name_or_path=self.base_model_name_or_path)
            self.base_model.eval()
            
        def _sample_from_model(self, texts, min_words=55, truncate_ratio=0.5):
            # encode each text as a list of token ids
            texts = [t.split(' ') for t in texts]
            # otherwise the entire length exceeds the maximum length
            texts = [t if len(t) <= 1024 else t[:1024] for t in texts]
            texts = [' '.join(t[: int(len(t) * truncate_ratio)]) for t in texts]
            all_encoded = self.base_tokenizer(texts, 
                                              return_tensors="pt", 
                                              padding=True,
                                              truncation=True,
                                              ).to(self.base_model.device)

            decoded = ['' for _ in range(len(texts))]
            # sample from the model until we get a sample with at least min_words words for each example
            # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
            tries = 0
            m = 0
            while m < min_words:
                if tries != 0:
                    print()
                    print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

                sampling_kwargs = {'temperature': self.temperature}
                min_length = 150
                outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_length=1024, do_sample=True,
                                                **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id,
                                                eos_token_id=self.base_tokenizer.eos_token_id)
                decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                m = min(len(x.split()) for x in decoded)
                tries += 1

            return decoded

        def generate_samples(self, raw_data, batch_size):
            # trim to shorter length
            def _trim_to_shorter_length(texta, textb):
                # truncate to shorter of o and s
                shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
                texta = ' '.join(texta.split(' ')[:shorter_length])
                textb = ' '.join(textb.split(' ')[:shorter_length])
                return texta, textb

            def _truncate_to_substring(text, substring, idx_occurrence):
                # truncate everything after the idx_occurrence occurrence of substring
                assert idx_occurrence > 0, 'idx_occurrence must be > 0'
                idx = -1
                for _ in range(idx_occurrence):
                    idx = text.find(substring, idx + 1)
                    if idx == -1:
                        return text
                return text[:idx]

            data = {
                "original": [],
                "sampled": [],
            }

            assert len(raw_data) % batch_size == 0

            for batch in range(len(raw_data) // batch_size):
                # print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
                original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
                sampled_text = self._sample_from_model(original_text, min_words=55, truncate_ratio=self.truncate_ratio)

                for o, s in zip(original_text, sampled_text):
                    o, s = _trim_to_shorter_length(o, s)

                    # add to the data
                    data["original"].append(o)
                    data["sampled"].append(s)

            return data

    def get_likelihood(self, logits, labels, pad_index):
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        mask = labels != pad_index
        log_likelihood = (log_likelihood * mask).sum(dim=1) / mask.sum(dim=1)
        return log_likelihood.squeeze(-1)

    def get_log_prob(self, text):
        tokenized = self.sampler.base_tokenizer(text, 
                                                return_tensors="pt", 
                                                padding=True,
                                                truncation=True
                                                ).to(self.sampler.base_model.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.sampler.base_model(**tokenized).logits[:, :-1]
            return self.get_likelihood(logits_score, labels, self.sampler.base_tokenizer.pad_token_id)

    def get_log_probs(self, texts):
        batch_size = self.sampler.batch_size
        batch_lprobs = []
        for batch in range(len(texts) // batch_size):
            tokenized = self.sampler.base_tokenizer(texts[batch * batch_size:(batch + 1) * batch_size],
                                                     return_tensors="pt", 
                                                     padding=True,
                                                     truncation=True,
                                                     ).to(self.sampler.base_model.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = self.sampler.base_model(**tokenized).logits[:, :-1]
                lprobs = self.get_likelihood(logits_score, labels, self.sampler.base_tokenizer.pad_token_id)
                batch_lprobs.append(lprobs)
        return torch.cat(batch_lprobs, dim=0)

    def get_regen_samples(self, text):
        data = [text] * self.sampler.regen_number
        data = self.sampler.generate_samples(data, batch_size=self.sampler.batch_size)
        return data['sampled']

    def get_dna_gpt_score(self, text):
        lprob = self.get_log_prob(text)
        regens = self.get_regen_samples(text)
        lprob_regens = self.get_log_probs(regens)
        wscore = lprob[0] - lprob_regens.mean()
        return wscore.item()
    
    def find_threshold(self, train_scores, train_labels):
        # Sort scores to get possible threshold values
        print(f"Finding best threshold for {self.mode}...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_accuracy = 0
        if self.mode == "low-fpr":
            scores = train_scores
            fpr, tpr, roc_thresholds = roc_curve(train_labels, scores)
            # Find the threshold where FPR is closest to 1%
            target_fpr = 0.01
            idx = np.where(fpr <= target_fpr)[0][-1]  # Closest index where FPR <= 1%
            self.threshold = roc_thresholds[idx]

        elif self.mode == "accuracy":
            for t in thresholds:
                predictions = train_scores > t
                accuracy = f1_score(train_labels, predictions)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = t

            self.threshold = best_threshold
        return best_threshold, best_accuracy

    def change_mode(self, mode):
        if mode not in ["low-fpr", "accuracy"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def detect(self, text, label, config):
        predictions = []
        for idx in tqdm(range(len(text)), desc="Detecting"):
            dna_gpt_score = self.get_dna_gpt_score(text[idx])
            predictions.append(dna_gpt_score)
        return predictions
        