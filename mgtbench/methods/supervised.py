import numpy as np
import transformers
import torch
from tqdm import tqdm
from ..utils import timeit, cal_metrics
from torch.utils.data import DataLoader
from transformers import AdamW
from ..auto import BaseDetector
from ..loading import load_pretrained_supervise
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SupervisedDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        self.model = kargs.get('model', None)
        self.tokenizer = kargs.get('tokenizer', None)
        if not self.model or not  self.tokenizer:
            model_name_or_path = kargs.get('model_name_or_path', None)
            if not model_name_or_path :
                raise ValueError('You should pass the model_name_or_path or a model instance, but none is given')
            quantitize_bit = kargs.get('load_in_k_bit', None)
            self.model, self.tokenizer = load_pretrained_supervise(model_name_or_path, quantitize_bit)
        if not isinstance(self.model, PreTrainedModel) or not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise ValueError('Expect PreTrainedModel, PreTrainedTokenizer, got', type(self.model), type(self.tokenizer))
        if ("state_dict_path" in kargs) and ("state_dict_key" in kargs):
            self.model.load_state_dict(
                torch.load(kargs["state_dict_path"],map_location='cpu')[kargs["state_dict_key"]])
        
    def detect(self, text, **kargs):
        result = []
        pos_bit=0
        if not isinstance(text, list):
            text = [text]
        for batch in tqdm(DataLoader(text)):
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=512,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                result.append(self.model(**tokenized).logits.softmax(-1)[:, pos_bit].item())
        # print(result)
        return result if isinstance(text, list) else result[0]
    
    def finetune(self, data, config):
        batch_size = config.batch_size
        num_epochs = config.epoch
        save_path = config.save_path
        if config.pos_bit == 0:
            train_label = [1 if _ == 0 else 0 for _ in train_label]
            test_label = [1 if _ == 0 else 0 for _ in test_label]

        train_encodings = self.tokenizer(data['text'], truncation=True, padding=True)
        train_dataset = CustomDataset(train_encodings, train_label)

        self.model.train()
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                running_loss += loss.item()
                optimizer.step()
            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
        self.model.eval()
        if config.need_save:
            self.model.save_pretrained(f'finetuned/{self.name}')


@timeit
def run_supervised_experiment(
        data,
        model,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=3,
        save_path=None,
        **kwargs):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels=num_labels,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model, cache_dir=cache_dir)

    if ("state_dict_path" in kwargs) and ("state_dict_key" in kwargs):
        detector.load_state_dict(
            torch.load(
                kwargs["state_dict_path"],
                map_location='cpu')[
                kwargs["state_dict_key"]])

    if finetune:
        fine_tune_model(detector, tokenizer, data, batch_size,
                        DEVICE, pos_bit, num_labels, epochs=epochs)
        if save_path:
            detector.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    # detector.save_pretrained(".cache/lm-d-xxx", from_pt=True)

    if num_labels == 2:
        train_preds = get_supervised_model_prediction(
            detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
    else:
        train_preds = get_supervised_model_prediction_multi_classes(
            detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction_multi_classes(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

    predictions = {
        'train': train_preds,
        'test': test_preds,
    }
    y_train_pred_prob = train_preds
    y_train_pred = [round(_) for _ in y_train_pred_prob]
    y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, recall_train, f1_train, auc_train = train_res
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res
    print(f"{model} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(f"{model} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        }
    }


def run_supervised_experiment_multi_test_length(
        data,
        model,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=3,
        save_path=None,
        lengths=[
            10,
            20,
            50,
            100,
            200,
            500,
            -1],
        **kwargs):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels=num_labels,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model, cache_dir=cache_dir)

    if ("state_dict_path" in kwargs) and ("state_dict_key" in kwargs):
        detector.load_state_dict(
            torch.load(
                kwargs["state_dict_path"],
                map_location='cpu')[
                kwargs["state_dict_key"]])

    if finetune:
        fine_tune_model(detector, tokenizer, data, batch_size,
                        DEVICE, pos_bit, num_labels, epochs=epochs)
        if save_path:
            detector.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    res = {}
    from mgtbench.utils import cut_length, cal_metrics
    for length in lengths:
        test_text = data['test']['text']
        test_text = [cut_length(_, length) for _ in test_text]
        test_label = data['test']['label']

        # detector.save_pretrained(".cache/lm-d-xxx", from_pt=True)

        if num_labels == 2:

            test_preds = get_supervised_model_prediction(
                detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
        else:
            test_preds = get_supervised_model_prediction_multi_classes(
                detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

        y_test_pred_prob = test_preds
        y_test_pred = [round(_) for _ in y_test_pred_prob]
        y_test = test_label

        test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res

        print(f"{model} {length} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
        res[length] = test_res
    # free GPU memory
    del detector
    torch.cuda.empty_cache()
    return res


def get_supervised_model_prediction(
        model,
        tokenizer,
        data,
        batch_size,
        DEVICE,
        pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt").to(DEVICE)
            preds.extend(model(**batch_data).logits.softmax(-1)
                         [:, pos_bit].tolist())
    return preds


def get_supervised_model_prediction_multi_classes(
        model, tokenizer, data, batch_size, DEVICE, pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt").to(DEVICE)
            preds.extend(torch.argmax(
                model(**batch_data).logits, dim=1).tolist())
    return preds


def fine_tune_model(
        model,
        tokenizer,
        data,
        batch_size,
        DEVICE,
        pos_bit=1,
        num_labels=2,
        epochs=3):

    # https://huggingface.co/transformers/v3.2.0/custom_datasets.html

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if pos_bit == 0 and num_labels == 2:
        train_label = [1 if _ == 0 else 0 for _ in train_label]
        test_label = [1 if _ == 0 else 0 for _ in test_label]

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_label)
    test_dataset = CustomDataset(test_encodings, test_label)

    model.train()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
    model.eval()
