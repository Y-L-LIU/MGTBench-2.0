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
        if 'input_ids' not in encodings:
            raise ValueError("Encodings must include 'input_ids'.")
        self.encodings = encodings
        self.labels = labels
        self.new_class = new_class

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if 'input_ids' not in item:
            raise ValueError(f"Missing 'input_ids' for sample {idx}")
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
        # logits = outputs.get("logits")
        
        # # Custom loss calculation (e.g., MSE or weighted cross-entropy)
        # loss_fn = torch.nn.CrossEntropyLoss()  # Example with weights for imbalance
        # loss = loss_fn(logits, labels)
        
        return (outputs.loss, outputs) if return_outputs else outputs.loss


class IncrementalModel(nn.Module):
    def __init__(self, model_name_or_path, kargs) -> None:
        super().__init__()
        self.pretrained, self.tokenizer = load_pretrained_supervise(model_name_or_path, kargs)
        # self.classifier = self.pretrained.classifier if hasattr(self.pretrained, "classifier") else self.pretrained.fc
        if hasattr(self.pretrained, "classifier"):
            self.classifier_attr = "classifier"
        elif hasattr(self.pretrained, "fc"):
            self.classifier_attr = "fc"
        else:
            raise AttributeError("The model does not have a recognizable classifier attribute.")
        self.n_classes = kargs.get('n_class', None)


    def forward(self, *args, **kwargs):
        return self.pretrained(*args, **kwargs)
    

    def increment_classes(self, new_classes):
        """Expand the classification head to accommodate new classes while retaining previous weights."""
        n = new_classes
        classifier = getattr(self.pretrained, self.classifier_attr)  # Get the classifier dynamically
        # print(classifier)
        in_features = classifier.in_features
        out_features = classifier.out_features
        weight = classifier.weight.data

        # Determine the new output features count
        new_out_features = out_features + n if self.n_classes > 0 else n

        # Update classifier with expanded output features
        new_classifier = nn.Linear(in_features, new_out_features, bias=True, dtype=torch.float16)
        new_classifier.to(self.pretrained.device)

        # Copy the old weights (from the existing classes) to the new classifier
        new_classifier.weight.data[:out_features] = weight  # Retain the previous weights

        # Initialize the new rows with the mean of the old weights
        mean_weight = weight.mean(dim=0, keepdim=True)  # Mean of the old weights across rows
        for i in range(out_features, new_out_features):
            new_classifier.weight.data[i] = mean_weight.squeeze(0)  # Set the new row to the mean        setattr(self.pretrained, 'num_labels', new_out_features)
        # Set the new classifier back to the model and update class counts
        self.pretrained.num_labels = new_out_features
        setattr(self.pretrained, self.classifier_attr, new_classifier)
        # self.classifier = new_classifier
        self.n_classes += n

def init_model(kargs):
    model_name_or_path = kargs.get('model_name_or_path', None)
    tmp = IncrementalModel(model_name_or_path, kargs)
    return tmp, tmp.tokenizer

class IncrementalDetector(BaseDetector):
    def __init__(self, name, **kargs):
        """
        Initialize with a specific base model (e.g., DistilBert, Bert).
        """
        super().__init__(name)
        self.model, self.tokenizer = init_model(kargs)
        # The number of initial classes

    def increment_classes(self, new_class):
        self.model.increment_classes(new_class)


    def detect(self, text, **kargs):
        disable_tqdm = kargs.get('disable_tqdm', False)
        result = []
        if not isinstance(text, list):
            text = [text]
        n_positions = self.model.pretrained.config.max_position_embeddings
        if n_positions<1000:
            n_positions=512
        else:
            n_positions=4096
        num_labels = self.model.pretrained.config.num_labels

        if num_labels == 2:
            pos_bit=1
            for batch in tqdm(DataLoader(text), disable=disable_tqdm):
                with torch.no_grad():
                    tokenized = self.tokenizer(
                        batch,
                        max_length=n_positions,
                        return_tensors="pt",
                        truncation = True
                    ).to(self.model.pretrained.device)
                    result.append(self.model(**tokenized).logits.softmax(-1)[:, pos_bit].item())
        else:
            for batch in tqdm(DataLoader(text), disable=disable_tqdm):
                with torch.no_grad():
                    tokenized = self.tokenizer(
                        batch,
                        max_length=n_positions,
                        return_tensors="pt",
                        truncation = True
                    ).to(self.model.pretrained.device)
                    result.append(torch.argmax(self.model(**tokenized).logits, dim=-1).item())

        return result if isinstance(text, list) else result[0]

    def get_dataset(self, stage_data):
        encodings = self.tokenizer(stage_data['text'], truncation=True, padding=True)
        unique_elements = set(stage_data['label'])
        num_newclass = len(unique_elements)
        dataset = ContinualDataset(encodings, stage_data['label'], num_newclass)
        return dataset


    def finetune(self, data, config):
        training_args = TrainingArguments(
            output_dir=config.save_path,              # Output directory
            num_train_epochs=config.epochs,           # Number of epochs
            per_device_train_batch_size=config.batch_size,  # Batch size
            save_strategy="epoch" if config.need_save else 'no', # Save after each epoch
            do_eval=True,
            evaluation_strategy='steps',
            logging_dir='./logs',                    # Directory for logs
            logging_steps=10,                       # Log every 100 steps
            weight_decay=0.01,                       # Weight decay
            learning_rate=config.lr,                      # Learning rate
            save_total_limit=2,                      # Limit to save only the best checkpoints
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            remove_unused_columns=False,
            label_names = ["labels"],
            # load_best_model_at_end=True if config.need_save else False  # Save best model
        )

        # prepare data for the first stage, each stage contains a continual dataset
        stages = data['train']
        eval_set = data['test']
        pre_lr = config.lr
        for idx, stage_data in enumerate(stages):
            train_dataset = self.get_dataset(stage_data)
            test_dataset = self.get_dataset(eval_set[idx])
            print(train_dataset[0].keys())
            if idx != 0:
                self.increment_classes(train_dataset.new_class)
            print(train_dataset.new_class, self.model.pretrained.num_labels,self.model.pretrained.classifier)

            trainer = ContinualTrainer(model=self.model,
                                    args=training_args,
                                    train_dataset=train_dataset,
                                    eval_dataset=test_dataset,
                                    tokenizer = self.tokenizer,
                                    optimizers=(AdamW(self.model.parameters(), lr=pre_lr), None)
                                    )
            trainer.train()
            pre_lr = config.lr / 8
            print(f'next lr is {pre_lr}')






