from typing import List, Literal, Optional, Tuple
from marshmallow import pre_dump
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
import copy
import random
from collections import defaultdict
from zmq import device
from ..loading.model_loader import load_pretrained_supervise
from ..auto import BaseDetector
from transformers import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import Trainer, TrainingArguments, AdamW
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
from torch.nn.functional import normalize, cosine_similarity
import time

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
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_mlp(X_train, y_train, input_dim, num_classes, hidden_dim=512, lr=1e-3, epochs=50):
    """Train an MLP classifier."""
    device = X_train.device
    model = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert labels to tensor
    y_train = y_train.to(device)

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model


class FewShotDetector(BaseDetector):
    def __init__(self, name, **kargs):
        """
        Initialize with a specific base model (e.g., DistilBert, Bert).
        """
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
        self.kshot = kargs.get('kshot',5)
        self.class_means = None


    def extract_embeddings(self, texts):
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k:inputs[k].to(self.model.device) for k in inputs.keys()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)  # Get hidden states from RoBERTa
            embeddings = outputs.hidden_states[-1][:, 0, :]  # Pool embeddings (mean over tokens)
            normalized_embeddings = normalize(embeddings, p=2, dim=1)  # L2 normalize
        return normalized_embeddings.float()
    
    def embed_samples(self, text, disable_tqdm=True):
        '''Extract the embeddings in result returned by sample_k_shot'''
        self.model.eval()  # Set model to evaluation mode
        all_embeds = []  # This will store the final result
        # Process each label group separately
        for label_texts in text:
            label_embeds = []  # Embeddings for each label's texts
            for batch in tqdm(DataLoader(label_texts), disable=disable_tqdm):
                embeddings = self.extract_embeddings(batch)  # Extract embeddings for this batch
                label_embeds.append(embeddings)  # Store embeddings for this batch
            all_embeds.append(label_embeds)  # Store embeddings for this label group
        return all_embeds

    def batch_embeddings(self, text, disable_tqdm=False):
        all_embeds = []
        for batch in tqdm(DataLoader(text), disable=disable_tqdm): 
            embeddings = self.extract_embeddings(batch)
            all_embeds.append(embeddings.cpu())
        return torch.cat(all_embeds, dim=0)

    def detect(self, text, **kargs):
        raise NotImplementedError('')

    # Step 1: Extract features for each sample
    def sample_k_shot(self, data, k):
        """Sample k examples per label from the dataset."""
        import time
        random.seed(time.time())
        text_by_label = defaultdict(list)
        
        # Organize data by label
        for text, label in zip(data['text'], data['label']):
            text_by_label[label].append(text)
        
        # Sample K examples per label
        sampled_texts, sampled_labels = [], []
        for label, texts in text_by_label.items():
            
            sampled = random.sample(texts, min(k, len(texts)))  # Ensure we don't oversample
            sampled_texts.append(sampled)
            sampled_labels.append([label] * len(sampled))
        
        return {'text': sampled_texts, 'label': sampled_labels}
    
    def construct_prototype(self, embeddings, labels):
        class_means = {}
        class_covs = {}
        
        # Iterate over each label group (each class)
        for label_list, embed_list in zip(labels, embeddings):
            # Stack embeddings within each label group (list of tensors to one tensor)
            class_embeds = torch.cat(embed_list, dim=0)  # Concatenate the embeddings across all batches for this label
            
            # Compute the mean for each class (mean across all embeddings for a particular class)
            class_means[label_list[0]] = class_embeds.mean(dim=0).cpu()  # Assuming the label is the same for all samples
            
            # Compute the covariance matrix for the class embeddings
            class_covs[label_list[0]] = torch.cov(class_embeds.T).cpu()  # Compute covariance for each class
        
        self.class_means = class_means
        self.class_covs = class_covs
        return class_means

    def compute_class_weights(self, dataset):
        labels = np.array(dataset.labels)
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.tensor(class_weights, dtype=torch.float)

    def update_shot(self, k):
        few_shot_data = self.sample_k_shot(self.all_data, k)
        embs = self.batch_embeddings(few_shot_data['text'])
        self.class_mean = self.construct_prototype(embs, torch.tensor(few_shot_data['label']))


class BaselineDetector(FewShotDetector):
    def __init__(self, name, **kargs):
        super().__init__(name, **kargs)

    def predict_with_cosine_similarity(self, test_embedding):
        test_embedding = test_embedding.unsqueeze(0) if test_embedding.dim() == 1 else test_embedding  # Ensure (1, hidden_dim)
        similarities = {
            label: cosine_similarity(test_embedding.cpu(), class_mean.unsqueeze(0), dim=1).item()
            for label, class_mean in self.class_means.items()
        }
        return max(similarities, key=similarities.get)  # Return label with highest similarity

    
    def detect(self, text, **kargs):
        disable_tqdm = kargs.get('disable_tqdm', False)
        result = []
        if not isinstance(text, list):
            text = [text]
        if not self.class_means:
            raise ValueError('Class prototype unitialized.')
        all_embeds = self.batch_embeddings(text, disable_tqdm)
        result = []
        for v in all_embeds:
            result.append(self.predict_with_cosine_similarity(v))
        return result if isinstance(text, list) else result[0]
    
    def finetune(self, data, config):
        self.kshot = config.kshot
        if config.need_finetune:
            training_args = TrainingArguments(
                output_dir=config.save_path,              # Output directory
                num_train_epochs=config.epochs,           # Number of epochs
                per_device_train_batch_size=config.batch_size,  # Batch size
                save_strategy="epoch" if config.need_save else 'no', # Save after each epoch
                do_eval=False,
                evaluation_strategy='no',
                logging_dir='./logs',                    # Directory for logs
                logging_steps=50,                       # Log every 100 steps
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
            train_encodings = self.tokenizer(stages[0]['text'], truncation=True, padding=True)
            train_dataset = CustomDataset(train_encodings, stages[0]['label'])
            
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
        all_data = {'text': list(data['train'][0]['text'])+list(data['train'][1]['text']),
                        'label': list(data['train'][0]['label'])+list(data['train'][1]['label'])}
        self.all_data = all_data
        examples = self.sample_k_shot(all_data, self.kshot)
        embeds = self.embed_samples(examples['text'])
        self.construct_prototype(embeds, examples['label'])


class GenerateDetector(FewShotDetector):
    def __init__(self, name, **kargs):
        super().__init__(name, **kargs)
        self.classifier = None

    def distribution_calibration(self, query, base_means, base_cov, k, alpha=0.21):
        """
        Perform distribution calibration using torch while accommodating dictionary-based class means and covariances.
        
        Args:
            query (torch.Tensor): Query embedding of shape [feature_dim].
            base_means (dict): Dictionary {label: tensor} of class mean embeddings.
            base_cov (dict): Dictionary {label: tensor} of class covariance matrices.
            k (int): Number of nearest class means to consider.
            alpha (float): Calibration adjustment factor.

        Returns:
            calibrated_mean (torch.Tensor): Adjusted mean of shape [feature_dim].
            calibrated_cov (torch.Tensor): Adjusted covariance of shape [feature_dim, feature_dim].
        """
        # Convert dictionary values to tensors
        class_labels = list(base_means.keys())  # Store class labels
        means_tensor = torch.stack([base_means[label] for label in class_labels]).to(query.device)  # [num_classes, feature_dim]
        covs_tensor = torch.stack([base_cov[label] for label in class_labels])  # [num_classes, feature_dim, feature_dim]

        # Compute L2 distances between query and each class mean
        dist = torch.norm(means_tensor - query.unsqueeze(0), dim=1)  # [num_classes]

        # Find the k nearest class means
        _, index = torch.topk(dist, k, largest=False)  # Indices of k closest class means

        # Retrieve the corresponding means and covariances
        selected_labels = [class_labels[i] for i in index.tolist()]
        selected_means = torch.stack([base_means[label] for label in selected_labels]).to(query.device)  # [k, feature_dim]
        selected_covs = torch.stack([base_cov[label] for label in selected_labels]).to(query.device)  # [k, feature_dim, feature_dim]
        
        # Compute the calibrated mean
        mean = torch.cat([selected_means, query.unsqueeze(0)], dim=0)  # [k+1, feature_dim]
        calibrated_mean = mean.mean(dim=0)  # [feature_dim]
        # Compute the calibrated covariance
        calibrated_cov = selected_covs.mean(dim=0) + alpha * torch.eye(selected_covs.shape[-1], device=query.device)  # [feature_dim, feature_dim]

        return calibrated_mean, calibrated_cov

    def data_augment(self, support_data, support_label, n_ways, samples = 750):
        """
        Use torch to augment the data with distribution calibration.
        """
        self.beta = 0.5
        sampled_data = []
        sampled_label = []
        num_sampled = int(samples / self.kshot)  # Number of samples per class

        for i in range(n_ways):
            # Flatten the list of tensors into a single tensor
            class_data = torch.cat(support_data[i], dim=0)  # [num_samples, feature_dim]
            class_label = support_label[i][0]  # Use the first label (all should be the same)

            # Apply power transformation
            class_data = torch.relu(class_data) ** self.beta  
            for i in range(self.kshot):
                # Perform distribution calibration
                mean, cov = self.distribution_calibration(class_data[i], 
                                                        self.class_means, 
                                                        self.class_covs, 
                                                        k=2)
                # print(mean)
                # Sample new data from the calibrated distribution
                dist = torch.distributions.MultivariateNormal(mean.to(dtype=torch.float32), cov.to(dtype=torch.float32))

                sampled_data_tensor = dist.sample((num_sampled,))  # Generate samples

                sampled_data.append(sampled_data_tensor)
                sampled_label.extend([class_label] * num_sampled)

        # Concatenate the original and augmented data
        X_aug = torch.cat([torch.cat([torch.cat(d, dim=0) for d in support_data]), torch.cat(sampled_data)], dim=0)
        Y_aug = torch.cat([torch.tensor([l for labels in support_label for l in labels]), torch.tensor(sampled_label)], dim=0)
        return X_aug, Y_aug

    def detect(self, text, **kargs):
        disable_tqdm = kargs.get('disable_tqdm', False)
        if not isinstance(text, list):
            text = [text]
        if not self.classifier:
            raise ValueError('Classifier not trained')
        all_embeds = self.batch_embeddings(text, disable_tqdm)
        all_embeds = torch.relu(all_embeds) ** self.beta  
        try:
            preds = self.classifier.predict(all_embeds.cpu())
        except:
            with torch.no_grad():
                logits = self.classifier(all_embeds.to(dtype=torch.float32).to(self.model.device))
                preds = torch.argmax(logits, dim=1).cpu()

        return preds if isinstance(text, list) else preds[0]

    def finetune(self, data, config):
        self.kshot = config.kshot
        if config.need_finetune:
            training_args = TrainingArguments(
                output_dir=config.save_path,              # Output directory
                num_train_epochs=config.epochs,           # Number of epochs
                per_device_train_batch_size=config.batch_size,  # Batch size
                save_strategy="epoch" if config.need_save else 'no', # Save after each epoch
                do_eval=False,
                evaluation_strategy='no',
                logging_dir='./logs',                    # Directory for logs
                logging_steps=50,                       # Log every 100 steps
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
            train_encodings = self.tokenizer(stages[0]['text'], truncation=True, padding=True)
            train_dataset = CustomDataset(train_encodings, stages[0]['label'])
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
        
        all_data = {'text': list(data['train'][0]['text'])+list(data['train'][1]['text']),
                        'label': list(data['train'][0]['label'])+list(data['train'][1]['label'])}
        examples = self.sample_k_shot(all_data,  self.kshot)
        embeds = self.embed_samples(examples['text'])
        self.construct_prototype(embeds, examples['label'])
        X_aug, Y_aug = self.data_augment(embeds, examples['label'], 6)
        if config.classifier == 'SVM':
            from sklearn.svm import SVC
            # Convert torch tensors to numpy
            X_train = X_aug.cpu().numpy()
            y_train = Y_aug.cpu().numpy()
            # Train an SVM classifier (use 'linear' or 'rbf' kernel)
            self.classifier = SVC(kernel='rbf', C=1.0, max_iter=1000)  
            self.classifier.fit(X_train, y_train)
            predicts = self.classifier.predict(X_train)
            acc = f1_score(y_train, predicts, average='macro')
            print(f"training Macro F1 Score: {acc:.4f}")


        elif config.classifier == 'Regression':
            from sklearn.linear_model import LogisticRegression
            X_train = X_aug.cpu().numpy()
            y_train = Y_aug.cpu().numpy()
            self.classifier = LogisticRegression(max_iter=1000).fit(X=X_train, y=y_train)
            predicts = self.classifier.predict(X_train)
            acc = f1_score(y_train, predicts, average='macro')
            print(f"training Macro F1 Score: {acc:.4f}")

        elif  config.classifier == 'MLP':
    # Convert labels to long tensor for classification
            X_train = X_aug  # Keep it as a torch tensor
            y_train = Y_aug.long()
            
            num_classes = len(torch.unique(y_train))
            self.classifier = train_mlp(X_train, y_train, input_dim=X_train.shape[1], num_classes=num_classes)

            # Get predictions
            with torch.no_grad():
                logits = self.classifier(X_train)
                predicts = torch.argmax(logits, dim=1)

            acc = f1_score(y_train.cpu().numpy(), predicts.cpu().numpy(), average='macro')
            print(f"training Macro F1 Score: {acc:.4f}")

class RNDetector(FewShotDetector):
    def __init__(self, name, **kargs):
        super().__init__(name, **kargs)
        if not isinstance(self.model, PreTrainedModel) or not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise ValueError("Expect PreTrainedModel and PreTrainedTokenizer")
        # Relation network
        self.relation_net = nn.Sequential(
            nn.Linear(self.model.config.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output similarity score
        ).to(self.model.device)
        # Hyperparameters
        self.num_classes = kargs.get('num_classes', 6)  # C-way
        self.kshot = kargs.get('kshot', 5)  # K-shot
        self.num_query = kargs.get('num_query', 15)  # Query samples per class
        self.beta = 0.5  # Power transformation factor for data augmentation
        self.optimizer = torch.optim.AdamW(
            list(self.relation_net.parameters()), lr=2e-3

        )

    def sample_episode(self, data):
        """
        Sample a few-shot episode (C-way, K-shot, N-query) from the dataset.

        Args:
            data (dict): A dictionary with 'text' and 'label' keys, each containing a list of samples.

        Returns:
            tuple: Contains support texts, query texts, support labels, and query labels.
        """
        # Combine text and label into a list of tuples
        combined_data = list(zip(data['text'], data['label']))

        # Organize data by label
        label_to_texts = defaultdict(list)
        for text, label in combined_data:
            label_to_texts[label].append(text)

        # Randomly sample C classes
        random.seed(time.time())
        sampled_classes = random.sample(list(label_to_texts.keys()), self.num_classes)

        support_set, query_set, support_labels, query_labels = [], [], [], []

        for label in sampled_classes:
            texts = label_to_texts[label]
            # Ensure there are enough samples to split into support and query sets
            if len(texts) < self.kshot + self.num_query:
                raise ValueError(f"Not enough samples for label {label} to perform k-shot and query sampling.")
            # Randomly sample without replacement
            sampled_texts = random.sample(texts, self.kshot + self.num_query)
            support_set.extend(sampled_texts[:self.kshot])
            query_set.extend(sampled_texts[self.kshot:])
            support_labels.extend([label] * self.kshot)
            query_labels.extend([label] * self.num_query)

        return support_set, query_set, support_labels, query_labels
    

    def finetune(self, data, config):
        """Train the model on a single episodic batch."""
        self.kshot = config.kshot
        if config.need_finetune:
            training_args = TrainingArguments(
                output_dir=config.save_path,              # Output directory
                num_train_epochs=config.epochs,           # Number of epochs
                per_device_train_batch_size=config.batch_size,  # Batch size
                save_strategy="epoch" if config.need_save else 'no', # Save after each epoch
                do_eval=False,
                evaluation_strategy='no',
                logging_dir='./logs',                    # Directory for logs
                logging_steps=50,                       # Log every 100 steps
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
            train_encodings = self.tokenizer(stages[0]['text'], truncation=True, padding=True)
            train_dataset = CustomDataset(train_encodings, stages[0]['label'])
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
        all_data = {'text': list(data['train'][0]['text'])+list(data['train'][1]['text']),
                        'label': list(data['train'][0]['label'])+list(data['train'][1]['label'])}

        self.model.train()
        self.relation_net.train()

        support_texts, query_texts, support_labels, query_labels = self.sample_episode(all_data)
        support_embeds = self.batch_embeddings(support_texts)
        query_embeds = self.batch_embeddings(query_texts)

        # Augment support set
        class_prototypes = {}
        for label in set(support_labels):
            class_indices = [i for i, lbl in enumerate(support_labels) if lbl == label]
            class_embeds = support_embeds[class_indices]
            class_prototypes[label] = class_embeds.mean(dim=0).cpu()
        self.class_means = class_prototypes
        relation_pairs, targets = [], []

        for i, query_embed in enumerate(query_embeds):
            for label, prototype in class_prototypes.items():
                relation_pairs.append(torch.cat([query_embed, prototype], dim=-1))
                targets.append(1 if query_labels[i] == label else 0)

        relation_pairs = torch.stack(relation_pairs).to(self.model.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.model.device)
        for _ in range(300):
            similarity_scores = self.relation_net(relation_pairs).squeeze()
            loss = nn.MSELoss()(similarity_scores, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if _%50==0:
                print(loss.item())


    def detect(self, text, disable_tqdm=True):
        """Predict the most similar class for an input text."""
        self.model.eval()
        self.relation_net.eval()
        if not isinstance(text, list):
            text = [text]
        new_embed = self.batch_embeddings(text)
        print(new_embed.shape, list(self.class_means.items())[0][1].shape)
        result = []
        for emb in new_embed:
            scores = {label: self.relation_net(torch.cat([emb, prototype], dim=-1).to(self.model.device)).item()
                    for label, prototype in self.class_means.items()}
            predicted_label = max(scores, key=scores.get)
            result.append(predicted_label)
        return result
