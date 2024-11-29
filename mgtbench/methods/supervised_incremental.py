from typing import List, Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
import copy
from ..loading.model_loader import load_pretrained_supervise
from ..auto import BaseDetector
from transformers import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import DistilBertPreTrainedModel, RobertaPreTrainedModel
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
        # Forward pass: get model outputs
        outputs = model(**inputs)
        logits = outputs.logits
        loss = outputs.loss
        labels = inputs.pop('labels', None)
        prev_model = model.get_prev()
        # Add regularization if there is a previous model
        if prev_model and self.is_in_train:
            with torch.no_grad():
                # Obtain logits from previous model without labels
                prev_logits = prev_model(**inputs).logits
            # Match dimensions for overlapping classes only
            num_shared_classes = prev_logits.size(-1)
            current_logits = logits[:, :num_shared_classes]

            func = F.kl_div 
            # Compute LwF regularization term (KL divergence)
            loss_reg = func(
                F.log_softmax(current_logits, dim=-1),
                F.softmax(prev_logits, dim=-1)           )
            # Add LwF regularization term to the task-specific loss
            alpha = model.lwf_reg  # Weighting factor for regularization, can be tuned
            loss = loss + alpha * loss_reg
        return (loss, outputs) if return_outputs else loss

class IncrementalModel(nn.Module):
    def __init__(self, model_name_or_path, kargs) -> None:
        super().__init__()
        self.pretrained, self.tokenizer = load_pretrained_supervise(model_name_or_path, kargs)
        self.lwf_reg = kargs.get('lwf_reg', 0.5)
        self.cache_size = kargs.get('cache_size', 100)
        print(f'lwf_reg is {self.lwf_reg}')
        # self.classifier = self.pretrained.classifier if hasattr(self.pretrained, "classifier") else self.pretrained.fc
        if hasattr(self.pretrained, "classifier"):
            self.classifier_attr = "classifier"
        elif hasattr(self.pretrained, "fc"):
            self.classifier_attr = "fc"
        else:
            raise AttributeError("The model does not have a recognizable classifier attribute.")
        self.n_classes = kargs.get('num_labels', None)
        self._prev_model = None
    
    def get_prev(self):
        return self._prev_model

    def set_head(self, new_classifier, new_out_features):
        setattr(self.pretrained, 'num_labels', new_out_features)
        # Set the new classifier back to the model and update class counts
        if isinstance(self.pretrained, DistilBertPreTrainedModel):
            setattr(self.pretrained, 'classifier', new_classifier)
        elif isinstance(self.pretrained, RobertaPreTrainedModel):
            setattr(self.pretrained.classifier, 'out_proj', new_classifier)

    def forward(self, *args, **kwargs):
        return self.pretrained(*args, **kwargs)
    
    def initialize_head(self, prev_classifier, new_out_features):
        in_features = prev_classifier.in_features
        out_features = prev_classifier.out_features
        weight = prev_classifier.weight.data
        # Update classifier with expanded output features
        new_classifier = nn.Linear(in_features, new_out_features, bias=False, dtype=torch.float16)
        new_classifier.to(self.pretrained.device)
        # kaiming_normal_(new_classifier.weight)  
        # Copy the old weights (from the existing classes) to the new classifier
        new_classifier.weight.data[:out_features] = weight  # Retain the previous weights

        # Initialize the new rows with the mean of the old weights
        mean_weight = copy.deepcopy(weight.mean(dim=0, keepdim=True))  # Mean of the old weights across rows
        for i in range(out_features, new_out_features):
            new_classifier.weight.data[i] = mean_weight.squeeze(0)  # Set the new row to the mean  
        new_classifier.weight.requires_grad = True
        return new_classifier     

    def increment_classes(self, new_classes, copy_prev=False):
        """Expand the classification head to accommodate new classes while retaining previous weights."""
        self._prev_model = self.pretrained
        if isinstance(self.pretrained, DistilBertPreTrainedModel):
            classifier = getattr(self.pretrained, 'classifier')
        elif isinstance(self.pretrained, RobertaPreTrainedModel):
            classifier = getattr(self.pretrained.classifier, 'out_proj')
        # print(classifier)
        new_out_features = classifier.out_features + new_classes if self.n_classes > 0 else new_classes
        new_classifier = self.initialize_head(classifier, new_out_features)
        # Determine the new output features count
        self.set_head(new_classifier, new_out_features)
        self.n_classes += new_classes

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

    def increment_classes(self, new_class, copy_prev=False):
        self.model.increment_classes(new_class, copy_prev)


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
        num_labels = self.model.pretrained.num_labels

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

    def get_dataset(self, stage_data, exampler=None,return_exampler=False):
        unique_elements = set(stage_data['label'])
        num_newclass = len(unique_elements)
        if exampler:
            stage_data['text'] = list(stage_data['text']) + list(exampler['text'])
            stage_data['label'] = list(stage_data['label']) + list(exampler['label'])
        encodings = self.tokenizer(stage_data['text'], truncation=True, padding=True)
        if return_exampler and self.model.cache_size!=0:
            print('construct the exampler for current class')
            exampler_idx = self.construct_exampler(stage_data, cache_size=self.model.cache_size)
            exampler = {'text':np.array(stage_data['text'])[exampler_idx], 'label':np.array(stage_data['label'])[exampler_idx]}
            print(f'Get exampler of {len(exampler_idx)} training data')

        dataset = ContinualDataset(encodings, stage_data['label'], num_newclass)
        return dataset, exampler 


    # Step 1: Extract features for each sample
    def construct_exampler(self, stage_data, cache_size=100):
        features = []
        labels = []
        print(len(stage_data['text']), len(stage_data['label']))
        for data in tqdm(stage_data['text']):
            encoding = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                # print(data, dataset[0])
                # encoding = {'input_ids':data.input_ids.to('cuda'), 
                #             'attention_mask':data.attention_mask.to('cuda')}
                outputs = self.model(**encoding.to('cuda'), output_hidden_states=True) 
                cls_embedding = outputs.hidden_states[-1][:, 0, :]
            features.append(cls_embedding.cpu().squeeze().numpy())
        labels = stage_data['label']
        features = np.array(features)
        from sklearn.metrics.pairwise import cosine_distances
        class_means = {}
        for label in np.unique(labels):
            class_features = features[labels == label]
            class_mean = np.mean(class_features, axis=0)
            class_means[label] = class_mean

        # Step 3: Compute distances to the class mean for each sample
        class_top_100 = []

        for label in np.unique(labels):
            # Get the indices and embeddings of samples in the current class
            class_indices = np.where(labels == label)[0]
            class_embeddings = features[class_indices]
            
            # Calculate distance from each sample to the class mean
            distances = []
            for i, embedding in zip(class_indices, class_embeddings):
                distance = cosine_distances([embedding], [class_means[label]])[0][0]
                distances.append((i, distance))
            
            # Sort by distance and select top-100 closest samples
            distances.sort(key=lambda x: x[1])  # Sort by distance
            top_100_for_class = distances[:cache_size]  # Select top-100 closest samples

            # Store the top-100 samples for the current class
            class_top_100.extend([i[0] for i in top_100_for_class])
        return class_top_100

    def finetune(self, data, config):
        training_args = TrainingArguments(
            output_dir=config.save_path,              # Output directory
            num_train_epochs=config.epochs,           # Number of epochs
            per_device_train_batch_size=config.batch_size,  # Batch size
            save_strategy="epoch" if config.need_save else 'no', # Save after each epoch
            do_eval=False,
            evaluation_strategy='no',
            logging_dir='./logs',                    # Directory for logs
            logging_steps=90,                       # Log every 100 steps
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
        exampler = None
        for idx, stage_data in enumerate(stages):
            train_dataset, exampler = self.get_dataset(stage_data,exampler=exampler,return_exampler=True)
            test_dataset = self.get_dataset(eval_set[idx], exampler=None,return_exampler=False)
            print(train_dataset[0].keys())
            if idx != 0:
                self.increment_classes(train_dataset.new_class, True)
                training_args.num_train_epochs = 1
            print(train_dataset.new_class, self.model.pretrained.num_labels,self.model.pretrained.classifier)

            trainer = ContinualTrainer(model=self.model,
                                    args=training_args,
                                    train_dataset=train_dataset,
                                    eval_dataset=test_dataset,
                                    tokenizer = self.tokenizer,
                                    optimizers=(AdamW(self.model.parameters(), lr=pre_lr), None)
                                    )
            # print([x for x in self.model.parameters()][-1], )
            trainer.train()
            factor = config.lr_factor
            # print([x for x in self.model.parameters()][-1], )
            pre_lr = config.lr/factor
            print(f'using {pre_lr}')






