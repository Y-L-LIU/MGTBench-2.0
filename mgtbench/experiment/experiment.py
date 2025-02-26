from ..auto import BaseExperiment, BaseDetector, DetectOutput
from ..methods import MetricBasedDetector, PerturbBasedDetector, SupervisedDetector, DemasqDetector, IncrementalDetector
import torch
import numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass, fields, asdict
from tqdm import tqdm
from ..loading.model_loader import load_pretrained_supervise


class ThresholdExperiment(BaseExperiment):
    _ALLOWED_detector = ['ll', 'rank', 'LRR', 'rank_GLTR', 'entropy', 'Binoculars']

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, MetricBasedDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
    
    def launch(self, **config):
        if not self.loaded:
            raise RuntimeError('You should load the data first, call load_data.')
        print('Calculate result for each data point')
        predict_list = self.predict(**config)
        final_output = []
        for detector_predict in predict_list:
            if len(detector_predict['train_pred']) == 2:
                train_metric1 = self.cal_metrics(*detector_predict['train_pred'][0])
                test_metric1 = self.cal_metrics(*detector_predict['test_pred'][0])
                train_metric2 = self.cal_metrics(*detector_predict['train_pred'][1])
                test_metric2 = self.cal_metrics(*detector_predict['test_pred'][1])
                final_output.append(DetectOutput(
                    name = 'threshold',
                    train = train_metric1,
                    test = test_metric1
                ))
                final_output.append(DetectOutput(
                    name = 'logistic',
                    train = train_metric2,
                    test = test_metric2
                ))
            else:
                train_metric = self.cal_metrics(*detector_predict['train_pred'])
                test_metric = self.cal_metrics(*detector_predict['test_pred'])
                final_output.append(DetectOutput(
                    name = 'logistic',
                    train = train_metric,
                    test = test_metric
                ))
        return final_output 
        
    def predict(self, **config):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for this experiment')
                continue
            if detector.name in ['rank_GLTR']:
                print('Predict training data')
                x_train, y_train = detector.detect(self.train_text), self.train_label
                x_train = np.array(x_train)
                y_train = np.array(y_train)
                print('Predict testing data')
                x_test, y_test = detector.detect(self.test_text), self.test_label
                x_test = np.array(x_test)
                y_test = np.array(y_test)
            else:
                print('Predict training data')
                x_train, y_train = self.data_prepare(detector.detect(self.train_text), self.train_label)
                print('Predict testing data')
                x_test, y_test = self.data_prepare(detector.detect(self.test_text), self.test_label)
                
            print('Run classification for results')
            # criterion = config.get('criterion', 'logistic')
            # assert criterion in ['logistic', 'threshold']
            # print('Prediction criterion:', criterion)

            # if criterion == 'threshold':
            if detector.name in ['Binoculars', 'rank', 'll', 'LRR', 'entropy']:
                print('Using threshold criterion')
                detector.find_threshold(x_train, y_train)
                if detector.name in ['rank', 'LRR', 'entropy', 'Binoculars']:
                    y_train_preds = [x < detector.threshold for x in x_train]
                    y_test_preds = [x < detector.threshold for x in x_test]
                    train_result1 = y_train, y_train_preds, -1 * x_train # human has higher score
                    test_result1 = y_test, y_test_preds, -1 * x_test

                elif detector.name in ['ll']:
                    y_train_preds = [x > detector.threshold for x in x_train]
                    y_test_preds = [x > detector.threshold for x in x_test]
                    train_result1 = y_train, y_train_preds, x_train
                    test_result1 = y_test, y_test_preds, x_test
                
                # logisitc regression
                print('Using logistic regression')
                clf = LogisticRegression(random_state=0).fit(np.clip(x_train, -1e10, 1e10), y_train)
                train_result2 = self.run_clf(clf, x_train, y_train)
                test_result2 = self.run_clf(clf, x_test, y_test)

                predict_list.append({'train_pred':(train_result1, train_result2),
                                        'test_pred':(test_result1, test_result2)})
            else:
                clf = LogisticRegression(random_state=0).fit(x_train, y_train)
                train_result = self.run_clf(clf, x_train, y_train)
                test_result = self.run_clf(clf, x_test, y_test)

                predict_list.append({'train_pred':train_result,
                                 'test_pred':test_result})
            
        return predict_list

@dataclass
class PerturbConfig:
    span_length:int = 2
    buffer_size:int = 1
    mask_top_p:float = 1 
    pct_words_masked:float = 0.3
    DEVICE:int = 0
    random_fills:bool = False
    random_fills_tokens:bool = False
    n_perturbation_rounds:int = 1
    n_perturbations:int = 10
    criterion_score:str = 'z'
    seed: int = 0

    def update(self, kargs):
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])


class PerturbExperiment(BaseExperiment):
    '''
    class PerturbConfig:
        span_length:int = 2
        buffer_size:int = 1
        mask_top_p:float = 1 
        pct_words_masked:float = 0.3
        DEVICE:int = 0
        random_fills:bool = False
        random_fills_tokens:bool = False
        n_perturbation_rounds:int = 1
        n_perturbations:int = 10
        criterion:str = 'd'
    '''
    _ALLOWED_detector = ['detectGPT', 'NPR', 'fast-detectGPT', 'DNA-GPT']
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, PerturbBasedDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.perturb_config= PerturbConfig()

    def launch(self, **config):
        if not self.loaded:
            raise RuntimeError('You should load the data first, call load_data.')
        print('Calculate result for each data point')
        predict_list = self.predict(**config)
        final_output = []
        for detector_predict in predict_list:
            if len(detector_predict['train_pred']) == 2:
                train_metric1 = self.cal_metrics(*detector_predict['train_pred'][0])
                test_metric1 = self.cal_metrics(*detector_predict['test_pred'][0])
                train_metric2 = self.cal_metrics(*detector_predict['train_pred'][1])
                test_metric2 = self.cal_metrics(*detector_predict['test_pred'][1])
                final_output.append(DetectOutput(
                    name = 'threshold',
                    train = train_metric1,
                    test = test_metric1
                ))
                final_output.append(DetectOutput(
                    name = 'logistic',
                    train = train_metric2,
                    test = test_metric2
                ))
            else:
                train_metric = self.cal_metrics(*detector_predict['train_pred'])
                test_metric = self.cal_metrics(*detector_predict['test_pred'])
                final_output.append(DetectOutput(
                    name = '',
                    train = train_metric,
                    test = test_metric
                ))
        return final_output
    

    def predict(self, **kargs):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            # print(kargs)

            self.perturb_config.update(kargs)
            print('Predict training data')
            x_train, y_train = self.data_prepare(detector.detect(self.train_text, self.train_label, self.perturb_config),self.train_label)
            print('Predict testing data')
            x_test, y_test   = self.data_prepare(detector.detect(self.test_text, self.test_label, self.perturb_config), self.test_label)
            print('Run classification for results')

            # criterion = kargs.get('criterion', 'logistic')
            # assert criterion in ['logistic', 'threshold']
            # print('Prediction criterion:', criterion)

            # if criterion == 'threshold':
            if detector.name in ['NPR', 'fast-detectGPT', 'DNA-GPT', 'detectGPT']:
                print('Using threshold criterion')
                detector.find_threshold(x_train, y_train)
                y_train_preds = [x > detector.threshold for x in x_train]
                y_test_preds = [x > detector.threshold for x in x_test]
                train_result1 = y_train, y_train_preds, x_train
                test_result1 = y_test, y_test_preds, x_test

                # logisitc regression
                print('Using logistic regression')
                clf = LogisticRegression(random_state=0).fit(x_train, y_train)
                train_result2 = self.run_clf(clf, x_train, y_train)
                test_result2 = self.run_clf(clf, x_test, y_test)

                predict_list.append({'train_pred':(train_result1, train_result2),
                                    'test_pred':(test_result1, test_result2)})

            else:
                clf = LogisticRegression(random_state=0).fit(x_train, y_train)
                train_result = self.run_clf(clf, x_train, y_train)
                test_result = self.run_clf(clf, x_test, y_test)

                predict_list.append({'train_pred':train_result,
                                    'test_pred':test_result})
            
        return predict_list


@dataclass
class SupervisedConfig:
    need_finetune:bool=False
    name:str=''
    need_save:bool=True
    batch_size:int=16
    pos_bit:int=1
    epochs:int=3
    save_path:str='finetuned/'
    gradient_accumulation_steps:int=1
    lr:float=5e-6

    def update(self, kargs):
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])

class SupervisedExperiment(BaseExperiment):
    '''
    class SupervisedConfig:
        need_finetune:bool=False
        need_save:bool=True
        batch_size:int=16
        pos_bit:int=1
        epochs:int=3
        save_path:str='finetuned/'
    '''
    _ALLOWED_detector = ['OpenAI-D', 'ConDA', 'ChatGPT-D', "LM-D", 'RADAR' ]
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, SupervisedDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.supervise_config= SupervisedConfig()

    def predict(self, **kargs):
        predict_list = []
        disable_tqdm = kargs.get('disable_tqdm', False)
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            self.supervise_config.update(kargs)
            if self.supervise_config.need_finetune:
                data_train = {'text':self.train_text, 'label':self.train_label}
                detector.finetune(data_train, self.supervise_config)
                print('Fine-tune finished')

            is_eval = kargs.get('eval', False)
            if is_eval:
                print('Predict testing data')
                test_preds, test_labels = self.data_prepare(detector.detect(self.test_text, disable_tqdm=disable_tqdm), self.test_label)
                print('Run classification for results')
                if detector.model.config.num_labels == 2:
                    y_test_pred = np.where(test_preds[:, 0] >= 0.5, 1, 0)
                    test_preds = [x for x in test_preds.flatten().tolist()]
                else:
                    y_test_pred = test_preds[:, 0]
                test_result = test_labels, y_test_pred, test_preds
                predict_list.append({'test_pred':test_result})

            else:
                print('Predict training data')
                train_preds, train_labels = self.data_prepare(detector.detect(self.train_text, disable_tqdm=disable_tqdm),self.train_label)
                print('Predict testing data')
                test_preds, test_labels = self.data_prepare(detector.detect(self.test_text, disable_tqdm=disable_tqdm), self.test_label)
                print('Run classification for results')

                if detector.model.config.num_labels == 2:
                    y_train_pred = np.where(train_preds[:, 0] >= 0.5, 1, 0)
                    y_test_pred = np.where(test_preds[:, 0] >= 0.5, 1, 0)
                    train_preds = [x for x in train_preds.flatten().tolist()]
                    test_preds = [x for x in test_preds.flatten().tolist()]
                else:
                    y_train_pred = train_preds[:, 0]
                    y_test_pred = test_preds[:, 0]

                train_result = train_labels, y_train_pred, train_preds
                test_result = test_labels, y_test_pred, test_preds
                # clf = LogisticRegression(random_state=0).fit(train_preds, train_labels)
                # train_result = self.run_clf(clf, train_preds, train_labels)
                # test_result = self.run_clf(clf, test_preds, test_labels)
                predict_list.append({'train_pred':train_result,
                                     'test_pred':test_result})
        return predict_list


@dataclass
class DemasqConfig:
    need_finetune:bool=True
    need_save:bool=True
    batch_size:int = 1
    save_path:str='model_weight/'
    epoch:int = 12
    
    def update(self, kargs):
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])


class DemasqExperiment(BaseExperiment):
    '''
    class DemasqConfig:
        need_finetune:bool=False
        batch_size:int = 1
        save_path:str='model_weight/'
        epoch:int = 12
    '''
    _ALLOWED_detector = ['demasq']
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, DemasqDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.config= DemasqConfig()

    def predict(self, **kargs):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            self.config.update(kargs)
            if self.config.need_finetune:
                data_train = {'text':self.train_text, 'label':self.train_label}
                detector.finetune(data_train, self.config)
            
            logits = detector.detect(self.train_text)
            preds = [1 if logit>0.5 else 0 for logit in logits]
            logits_t = detector.detect(self.test_text)
            preds_t = [1 if logit>0.5 else 0 for logit in logits_t]
            predict_list.append({'train_pred':(self.train_label, preds, logits),
                                 'test_pred':(self.test_label, preds_t, logits_t)})
        return predict_list

class GPTZeroExperiment(BaseExperiment):
    _ALLOWED_detector = ['GPTZero']
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, DemasqDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')

    def predict(self, **kargs):
        predict_list = []
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for _ALLOWED_detector')
                continue
            
            logits = detector.detect(self.train_text)
            preds = [1 if logit>0.5 else 0 for logit in logits]
            logits_t = detector.detect(self.test_text)
            preds_t = [1 if logit>0.5 else 0 for logit in logits_t]
            predict_list.append({'train_pred':(self.train_label, preds, logits),
                                 'test_pred':(self.test_label, preds_t, logits_t)})
        return predict_list


@dataclass
class IncrementalConfig:
    need_finetune:bool=False
    name:str=''
    need_save:bool=True
    batch_size:int=16
    pos_bit:int=1
    epochs:int=1
    save_path:str='finetuned/'
    gradient_accumulation_steps:int=1
    lr:float=5e-6
    lr_factor:int = 5

    def update(self, kargs):
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])


class IncrementalExperiment(BaseExperiment):
    '''
    class SupervisedConfig:
        need_finetune:bool=False
        need_save:bool=True
        batch_size:int=16
        pos_bit:int=1
        epochs:int=3
        save_path:str='finetuned/'
    '''
    _ALLOWED_detector = ['incremental' ]
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, IncrementalDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.supervise_config= IncrementalConfig()
        

    def load_data(self, data):
        self.loaded = True
        self.data = data
        self.train_text = data['train'][-1]['text']
        self.train_label = data['train'][-1]['label']
        self.test_text = data['test'][-1]['text']
        self.test_label = data['test'][-1]['label']

    def return_output(self, detector, pair=None, intermedia=None) :
        if not pair and not intermedia:
            raise ValueError('At least one text or intermedia should be given for predictioon ')
        if not intermedia:
            inter_preds, inter_labels = self.data_prepare(detector.detect(pair[0], disable_tqdm=True),pair[1])
        else:
            inter_preds, inter_labels = self.data_prepare(*intermedia)
        print('Run classification for results')
        if detector.model.pretrained.num_labels == 2:
            y_inter_preds = np.where(inter_preds[:, 0] >= 0.5, 1, 0)
            inter_preds = [x for x in inter_preds.flatten().tolist()]
        else:
            y_inter_preds = inter_preds[:, 0]
        test_result = inter_labels, y_inter_preds, inter_preds
        return test_result


    def predict(self, **kargs):
        predict_list = []
        disable_tqdm = kargs.get('disable_tqdm', False)
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            self.supervise_config.update(kargs)
            intermedia = None
            if self.supervise_config.need_finetune:
                intermedia = detector.finetune(self.data, self.supervise_config)
                print('Fine-tune finished')
            print(detector.model.use_bic)
            is_eval = kargs.get('eval', False)
            if is_eval:
                print('Predict testing data') 
                if intermedia:
                    predict_list.append({'intermedia_pred':self.return_output(detector, intermedia=intermedia)})
                predict_list.append({'test_pred':self.return_output(detector, pair=(self.test_text, self.test_label))})

            else:
                predict_list.append({'train_pred':self.return_output(detector, pair=(self.trian_text, self.train_label))})
                predict_list.append({'test_pred':self.return_output(detector, pair=(self.test_text, self.test_label))})
        return predict_list



class IncrementalThresholdExperiment(BaseExperiment):
    '''
    class SupervisedConfig:
        need_finetune:bool=False
        need_save:bool=True
        batch_size:int=16
        pos_bit:int=1
        epochs:int=3
        save_path:str='finetuned/'
    '''
    _ALLOWED_detector = ['ll', 'rank', 'LRR', 'rank_GLTR', 'entropy', 'Binoculars','fast-detectGPT']
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, IncrementalDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.model, self.tokenizer = load_pretrained_supervise('/data1/models/roberta-base', kargs)
        self.cache_size = kargs.get('cache_size', 0)
        print(self.cache_size)
        
        

    def load_data(self, data):
        self.loaded = True
        self.data = data
        self.train_text = data['train'][-1]['text']
        self.train_label = data['train'][-1]['label']
        self.test_text = data['test'][-1]['text']
        self.test_label = data['test'][-1]['label']

    def get_dataset(self, stage_data, exampler=None,return_exampler=False):
        if exampler:
            stage_data['text'] = list(stage_data['text']) + list(exampler['text'])
            stage_data['label'] = list(stage_data['label']) + list(exampler['label'])
        if return_exampler and self.cache_size!=0:
            print('construct the exampler for current class')
            exampler_idx = self.construct_exampler(stage_data, cache_size=self.cache_size)
            exampler = {'text':np.array(stage_data['text'])[exampler_idx], 'label':np.array(stage_data['label'])[exampler_idx]}
            print(f'Get exampler of {len(exampler_idx)} training data')
        # stage_data['text'] = stage_data['text'][:200]
        # stage_data['label'] = stage_data['label'][:200]
        return stage_data, exampler

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
    
    def increment_classes(self, detector, new_classes):
        clf = detector.classifier
        old_coef = clf.coef_
        old_intercept = clf.intercept_

        new_coef = np.zeros((new_classes, old_coef.shape[1]))
        new_intercept = np.zeros(new_classes)

        # Combine old and new parameters
        expanded_coef = np.vstack([old_coef, new_coef])
        expanded_intercept = np.hstack([old_intercept, new_intercept])

        # Set new parameters in a fresh Logistic Regression model
        clf.classes_ = np.arange(clf.classes_.size + new_classes)  # Update classes
        clf.coef_ = expanded_coef
        clf.intercept_ = expanded_intercept
        detector.classifier = clf


    def predict(self, **kargs):
        predict_list = []
        disable_tqdm = kargs.get('disable_tqdm', False)
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            stages = self.data['train']
            eval_set = self.data['test']
            exampler=None
            for idx, stage_data in enumerate(stages):
                train_dataset, exampler = self.get_dataset(stage_data,exampler=exampler,return_exampler=True)
                test_dataset,_ = self.get_dataset(eval_set[idx], exampler=None,return_exampler=False)
                if idx != 0:
                    unique_elements = set(stage_data['label'])
                    num_newclass = len(unique_elements)
                    self.increment_classes(detector, num_newclass)#check
                    print(detector.classifier.coef_.shape)

                if detector.name in ['rank_GLTR']:
                    print('Predict training data')
                    x_train, y_train = detector.detect(train_dataset['text']), train_dataset['label']
                    x_train = np.array(x_train)
                    y_train = np.array(y_train)
                    print('Predict testing data')
                    x_test, y_test = detector.detect(test_dataset['text']), test_dataset['label']
                    x_test = np.array(x_test)
                    y_test = np.array(y_test)
                else:
                    print('Predict training data')
                    print((train_dataset['text'][0]))
                    x_train, y_train = self.data_prepare(detector.detect(train_dataset['text']), train_dataset['label'])
                    print('Predict testing data')
                    x_test, y_test = self.data_prepare(detector.detect(test_dataset['text']), test_dataset['label'])
                detector.classifier.fit(x_train, y_train)
                test_result = self.run_clf(detector.classifier, x_test, y_test)
                predict_list.append({'test_pred':test_result})

        return predict_list


@dataclass
class FewShotConfig:
    need_finetune:bool=False
    name:str=''
    need_save:bool=True
    batch_size:int=16
    pos_bit:int=1
    epochs:int=1
    save_path:str='finetuned/'
    gradient_accumulation_steps:int=1
    lr:float=5e-6
    lr_factor:int = 5
    kshot:int = 5
    classifier:str = 'SVM'

    def update(self, kargs):
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])


class FewShotExperiment(BaseExperiment):
    '''
    class SupervisedConfig:
        need_finetune:bool=False
        need_save:bool=True
        batch_size:int=16
        pos_bit:int=1
        epochs:int=3
        save_path:str='finetuned/'
    '''
    _ALLOWED_detector = ['baseline', 'generate', 'rn']
    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, IncrementalDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        self.supervise_config= FewShotConfig()
        self.kshot = None
        

    def load_data(self, data):
        self.loaded = True
        self.data = data
        self.train_text = data['train'][-1]['text']
        self.train_label = data['train'][-1]['label']
        self.test_text = data['test'][-1]['text']
        self.test_label = data['test'][-1]['label']

    def return_output(self, detector, pair=None, intermedia=None) :
        if not pair and not intermedia:
            raise ValueError('At least one text or intermedia should be given for predictioon ')
        if not intermedia:
            inter_preds, inter_labels = self.data_prepare(detector.detect(pair[0], disable_tqdm=True),pair[1])
        else:
            inter_preds, inter_labels = self.data_prepare(*intermedia)
        print('Run classification for results')
        if detector.model.num_labels == 2:
            y_inter_preds = np.where(inter_preds[:, 0] >= 0.5, 1, 0)
            inter_preds = [x for x in inter_preds.flatten().tolist()]
        else:
            y_inter_preds = inter_preds[:, 0]
        test_result = inter_labels, y_inter_preds, inter_preds

        return test_result


    def predict(self, **kargs):
        predict_list = []
        disable_tqdm = kargs.get('disable_tqdm', False)
        for detector in self.detector:
            print(f'Running prediction of detector {detector.name}')
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, 'is not for it')
                continue
            self.supervise_config.update(kargs)
            # intermedia = None
            detector.finetune(self.data, self.supervise_config)
            print('Fine-tune finished')
            is_eval = kargs.get('eval', False)
            
            if is_eval:
                print('Predict testing data') 
                predict_list.append({'test_pred':self.return_output(detector, pair=(self.test_text, self.test_label))})

            else:
                predict_list.append({'train_pred':self.return_output(detector, pair=(self.trian_text, self.train_label))})
                predict_list.append({'test_pred':self.return_output(detector, pair=(self.test_text, self.test_label))})
        return predict_list