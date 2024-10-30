from typing import List, Literal, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..loading.model_loader import load_pretrained_supervise
from ..auto import BaseDetector
from transformers import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, AdamW
class ContinualDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, new_class):
        self.encodings = encodings
        self.labels = labels
        self.new_class = new_class

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ContinualTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels from inputs
        labels = inputs.get("labels")
        
        # Forward pass: get model outputs
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Custom loss calculation (e.g., MSE or weighted cross-entropy)
        loss_fn = torch.nn.CrossEntropyLoss()  # Example with weights for imbalance
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss



class IncrementalModel(nn.Module):
    def __init__(self, model_name_or_path, **kargs) -> None:
        super().__init__()
        self.model, self.tokenizer = load_pretrained_supervise(model_name_or_path, kargs)
        self.classifier = self.model.classifier if hasattr(self.model, "classifier") else self.model.fc
        if hasattr(self.model, "classifier"):
            self.classifier_attr = "classifier"
        elif hasattr(self.model, "fc"):
            self.classifier_attr = "fc"
        else:
            raise AttributeError("The model does not have a recognizable classifier attribute.")


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    

    def increment_classes(self, new_classes):
        """Expand the classification head to accommodate new classes while retaining previous weights."""
        n = len(new_classes)
        classifier = getattr(self.model, self.classifier_attr)  # Get the classifier dynamically
        in_features = classifier.in_features
        out_features = classifier.out_features
        weight = classifier.weight.data

        # Determine the new output features count
        new_out_features = out_features + n if self.n_classes > 0 else n

        # Update classifier with expanded output features
        new_classifier = nn.Linear(in_features, new_out_features, bias=False)
        kaiming_normal_(new_classifier.weight)  # Initialize new weights
        new_classifier.weight.data[:out_features] = weight  # Retain previous weights

        # Set the new classifier back to the model and update class counts
        setattr(self.model, self.classifier_attr, new_classifier)
        self.n_classes += n

def init_model(**kargs):
    model_name_or_path = kargs.get('model_name_or_path', None)
    tmp = IncrementalModel(model_name_or_path, kargs)
    return tmp, tmp.tokenizer

class IncrementalDetector(nn.Module, BaseDetector):
    def __init__(self, name, **kargs):
        """
        Initialize with a specific base model (e.g., DistilBert, Bert).
        """
        super().__init__()
        self.model, self.tokenizer = init_model(kargs)
        # The number of initial classes
        self.n_classes = kargs.get('n_class', None)

    def increment_classes(self, new_class):
        self.model.increment_classes(new_class)


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

        for batch in tqdm(DataLoader(text), disable=disable_tqdm):
            #TODO: check if two cases are consistent? (num_label=2 or more)
            with torch.no_grad():
                tokenized = self.tokenizer(
                    batch,
                    max_length=n_positions,
                    return_tensors="pt",
                    truncation = True
                ).to(self.model.device)
                result.append(torch.argmax(self.model(**tokenized).logits, dim=-1).item())
        return result if isinstance(text, list) else result[0]

    def finetune(self, data, config):
        if config.pos_bit == 0:
            data['label'] = [1 if label == 0 else 0 for label in data['label']]

        training_args = TrainingArguments(
            output_dir=config.save_path,              # Output directory
            num_train_epochs=config.epochs,           # Number of epochs
            per_device_train_batch_size=config.batch_size,  # Batch size
            save_strategy="epoch" if config.need_save else 'no', # Save after each epoch
            logging_dir='./logs',                    # Directory for logs
            logging_steps=50,                       # Log every 100 steps
            weight_decay=0.01,                       # Weight decay
            learning_rate=config.lr,                      # Learning rate
            save_total_limit=2,                      # Limit to save only the best checkpoints
            gradient_accumulation_steps=config.gradient_accumulation_steps
            # load_best_model_at_end=True if config.need_save else False  # Save best model
        )

        # prepare data for the first stage, each stage contains a continual dataset
        stages = []
        for stage_data in stages:
            train_encodings = self.tokenizer(stage_data['text'], truncation=True, padding=True)
            num_newclass = stage_data.get('new_class', None)
            if not num_newclass:
                assert(ValueError, 'dataset should contain new_class attribution')
            train_dataset = ContinualDataset(train_encodings, data['label'], num_newclass)
            self.increment_classes(num_newclass)
            trainer = ContinualTrainer(model=self.model,
                                    args=training_args,
                                    train_dataset=train_dataset)
            trainer.train()




