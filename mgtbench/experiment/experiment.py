from ..auto import BaseExperiment, BaseDetector, DetectOutput
from ..methods import MetricBasedDetector, PerturbBasedDetector, SupervisedDetector, DemasqDetector, IncrementalDetector
import torch
import numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass, fields, asdict


class ThresholdExperiment(BaseExperiment):
    _ALLOWED_detector = ['ll', 'rank', 'LRR', 'rank_GLTR', 'entropy', 'Binoculars']

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = [detector] if isinstance(detector, MetricBasedDetector) else detector
        if not self.detector:
            raise ValueError('You should pass a list of detector to an experiment')
        
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
            criterion = config.get('criterion', 'logistic')
            assert criterion in ['logistic', 'threshold']
            print('Prediction criterion:', criterion)

            if criterion == 'threshold':
                assert detector.name in ['Binoculars', 'rank', 'll', 'LRR', 'entropy']
                if detector.name in ['Binoculars']:
                    detector.find_threshold(x_train, y_train)
                    y_train_preds = [x < detector.threshold for x in x_train]
                    y_test_preds = [x < detector.threshold for x in x_test]
                    train_result = y_train, y_train_preds, -1 * x_train # for auc, binocular score is higher for human
                    test_result = y_test, y_test_preds, -1 * x_test

                elif detector.name in ['rank', 'll', 'LRR', 'entropy']:
                    detector.find_threshold(x_train, y_train)
                    if detector.name in ['rank', 'LRR', 'entropy']:
                        y_train_preds = [x < detector.threshold for x in x_train]
                        y_test_preds = [x < detector.threshold for x in x_test]
                        train_result = y_train, y_train_preds, -1 * x_train # human has higher score
                        test_result = y_test, y_test_preds, -1 * x_test

                    elif detector.name in ['ll']:
                        y_train_preds = [x > detector.threshold for x in x_train]
                        y_test_preds = [x > detector.threshold for x in x_test]
                        train_result = y_train, y_train_preds, x_train
                        test_result = y_test, y_test_preds, x_test
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

            criterion = kargs.get('criterion', 'logistic')
            assert criterion in ['logistic', 'threshold']
            print('Prediction criterion:', criterion)

            if criterion == 'threshold':
                if detector.name in ['NPR', 'fast-detectGPT', 'DNA-GPT', 'detectGPT']:
                    detector.find_threshold(x_train, y_train)
                    y_train_preds = [x > detector.threshold for x in x_train]
                    y_test_preds = [x > detector.threshold for x in x_test]
                    train_result = y_train, y_train_preds, x_train
                    test_result = y_test, y_test_preds, x_test
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
                detector.finetune(self.data, self.supervise_config)
                print('Fine-tune finished')

            is_eval = kargs.get('eval', False)
            if is_eval:
                print('Predict testing data')
                test_preds, test_labels = self.data_prepare(detector.detect(self.test_text, disable_tqdm=disable_tqdm), self.test_label)
                print('Run classification for results')
                if detector.model.pretrained.num_labels == 2:
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

                if detector.model.pretrained.num_labels == 2:
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
