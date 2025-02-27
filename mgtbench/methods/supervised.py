import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from ..auto import BaseDetector
from ..loading import load_pretrained_supervise
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import Trainer, TrainingArguments, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

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


class MetricTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass: get model outputs
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        labels = inputs.pop('labels', None)
        ce_loss = nn.CrossEntropyLoss()(logits, labels)
        # Get embeddings (last hidden state)
        embeddings = outputs.hidden_states[-1]  # Shape: (batch_size, seq_length, hidden_dim)
        embeddings = embeddings[:, 0, :]  # Take [CLS] token embedding (batch_size, hidden_dim)
        # Alternatively: embeddings = embeddings.mean(dim=1)

        # Compute custom loss
        loss_fn = CircleLoss() 
        npmc_loss = loss_fn(embeddings, labels)
        return (npmc_loss+ ce_loss, outputs) if return_outputs else npmc_loss+ ce_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, gamma=2, margin=0.25):
        super(CircleLoss, self).__init__()
        self.gamma = gamma
        self.margin = margin
        self.O_p = 1 + margin  # Upper bound for positive similarity
        self.O_n = -margin     # Lower bound for negative similarity
        self.Delta_p = 1 - margin  # Adjusted margin for positives
        self.Delta_n = margin      # Adjusted margin for negatives

    def forward(self, embeddings, labels):
        """
        Compute the Circle Loss.

        Args:
        embeddings: Tensor of shape (batch_size, hidden_dim) containing feature embeddings.
        labels: Tensor of shape (batch_size,) containing class labels.

        Returns:
        loss: Scalar tensor representing the loss value.
        """
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Pairwise similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # Shape: (batch_size, batch_size)

        # Create masks for positives and negatives
        labels = labels.view(-1, 1)
        positive_mask = labels.eq(labels.T)  # Same class -> True
        negative_mask = ~positive_mask       # Different class -> True

        # Collect positive and negative similarity scores dynamically
        positive_loss, negative_loss = 0.0, 0.0
        for i in range(sim_matrix.size(0)):  # Loop through batch
            S_p = sim_matrix[i][positive_mask[i]]  # Positives for sample i
            S_n = sim_matrix[i][negative_mask[i]]  # Negatives for sample i

            # Weighting for positive and negative samples
            alpha_p = F.relu(S_p - self.Delta_p)  # Positive weighting
            alpha_n = F.relu(self.Delta_n - S_n)  # Negative weighting

            # Circle loss components for current 
            positive_loss += torch.sum(alpha_p * torch.exp(-self.gamma * (S_p - self.O_p)))
            negative_loss += torch.sum(alpha_n * torch.exp(self.gamma * (S_n - self.O_n)))
            assert not torch.isnan(positive_loss).any(), f"{S_p} found in exponential computation"


        # Combine all losses
        loss = torch.log(1 + positive_loss + negative_loss)

        assert not torch.isnan(embeddings).any(), "NaN found in embeddings"
        assert not torch.isinf(embeddings).any(), "Inf found in embeddings"

        # Debugging for similarity matrix
        assert not torch.isnan(sim_matrix).any(), "NaN found in similarity matrix"
        assert not torch.isinf(sim_matrix).any(), "Inf found in similarity matrix"

        # Debugging for exponential computation
        assert not torch.isnan(positive_loss).any(), "NaN found in exponential computation"
        assert not torch.isnan(negative_loss).any(), "Inf found in exponential computation"
        return loss



class SupervisedDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        self.model = kargs.get('model', None)
        self.tokenizer = kargs.get('tokenizer', None)
        self.use_metric = kargs.get('use_metric', False)
        if not self.model or not  self.tokenizer:
            model_name_or_path = kargs.get('model_name_or_path', None)
            if not model_name_or_path :
                raise ValueError('You should pass the model_name_or_path or a model instance, but none is given')
            quantitize_bit = kargs.get('load_in_k_bit', None)
            self.model, self.tokenizer = load_pretrained_supervise(model_name_or_path, kargs,quantitize_bit)
        if not isinstance(self.model, PreTrainedModel) or not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise ValueError('Expect PreTrainedModel, PreTrainedTokenizer, got', type(self.model), type(self.tokenizer))
        
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
        # TODO: combine the two cases use inner loop if
        if num_labels == 2:
            pos_bit=1
            for batch in tqdm(DataLoader(text), disable=disable_tqdm):
                with torch.no_grad():
                    tokenized = self.tokenizer(
                        batch,
                        max_length=n_positions,
                        return_tensors="pt",
                        truncation = True
                    ).to(self.model.device)
                    result.append(self.model(**tokenized).logits.softmax(-1)[:, pos_bit].item())
        else:
            for batch in tqdm(DataLoader(text), disable=disable_tqdm):
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
        # Tokenize the data
        train_encodings = self.tokenizer(data['text'], truncation=True, padding=True)
        train_dataset = CustomDataset(train_encodings, data['label'])
        print(config.need_save)
        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=config.save_path,              # Output directory
            num_train_epochs=config.epochs,           # Number of epochs
            per_device_train_batch_size=config.batch_size,  # Batch size
            save_strategy="epoch" if config.need_save else 'no', # Save after each epoch
            logging_dir='./logs',                    # Directory for logs
            logging_steps=30,                       # Log every 100 steps
            weight_decay=0.01,                       # Weight decay
            learning_rate=config.lr,                      # Learning rate
            save_total_limit=2,                      # Limit to save only the best checkpoints
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            # lr_scheduler_type='cosine',
            # load_best_model_at_end=True if config.need_save else False  # Save best model
        )

        # Initialize the Trainer
        if self.use_metric:
            print('using metric trainer')
            trainer = MetricTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
                optimizers=(AdamW(self.model.parameters(), lr=config.lr), None)  # Optimizer, lr_scheduler
                # do not use torch.optim.AdamW, will cause nan
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
                optimizers=(AdamW(self.model.parameters(), lr=config.lr), None)  # Optimizer, lr_scheduler
                # do not use torch.optim.AdamW, will cause nan
            )


        # Train the model
        trainer.train()

        # Save the model if needed
        # if config.need_save:
        #     self.model.save_pretrained(f'{config.save_path}/{config.name}')