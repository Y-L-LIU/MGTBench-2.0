import numpy as np
import pandas as pd
import os
import csv
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from mgtbench.auto import Metric, DetectOutput

METHODS = ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR',
           'Binoculars', 'fast-detectGPT']

TOPICS = ['STEM', 'Humanities', 'Social_sciences']

LLMS = ['Moonshot', 'Mixtral', 'gpt35', 'Llama3', 'gpt-4omini']

ZEROSHOT_METHODS = ['ll', 'rank', 'rank_GLTR', 'entropy', 'LRR',
           'Binoculars',
           'fast-detectGPT', 
        #    'DNA-GPT',
        #    'detectGPT', 'NPR',
            ]

TRAIN_SIZE = 1000
TEST_SIZE = 2000

def run_clf(clf, x, y):
    # Clip extreme values
    x = np.clip(x, -1e10, 1e10)
    y_train_pred = clf.predict(x)
    y_train_pred_prob = clf.predict_proba(x)
    y_train_pred_prob = [_[1] for _ in y_train_pred_prob]
    return (y, y_train_pred, y_train_pred_prob)


def cal_metrics(label, pred_label, pred_posteriors) -> Metric:
    if max(set(label)) < 2:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        if sum(label) > 0 and sum(label) < len(label):
            auc = roc_auc_score(label, pred_posteriors)
        else:
            auc = -1.0
        return Metric(acc, precision, recall, f1, auc)
    else:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average='weighted')
        recall = recall_score(label, pred_label, average='weighted')
        f1 = f1_score(label, pred_label, average='weighted')
        auc = -1.0
        conf_m = confusion_matrix(label, pred_label)
        return Metric(acc, precision, recall, f1, auc, conf_m)


def train_logistic(source_x_train: np.array,
                    source_y_train: np.array,
                    target_x_train: np.array,
                    target_y_train: np.array,
                    target_x_test: np.array,
                    target_y_test: np.array,
                    mitigate_size: int
                    ):
    # Train the classifier
    assert len(target_x_train) > mitigate_size
    if mitigate_size == 0:
        x_train = source_x_train
        y_train = source_y_train
    mitigate_x_train = target_x_train[:mitigate_size]
    mitigate_y_train = target_y_train[:mitigate_size]
    x_train = np.concatenate([source_x_train, mitigate_x_train])
    y_train = np.concatenate([source_y_train, mitigate_y_train])
    clf = LogisticRegression(random_state=0).fit(np.clip(x_train, -1e10, 1e10), y_train)
    train_result = run_clf(clf, x_train, y_train)
    test_result = run_clf(clf, target_x_test, target_y_test)
    return train_result, test_result


def train_svm(method: str,
              source_x_train: np.array, 
              source_y_train: np.array,
              target_x_train: np.array,
              target_y_train: np.array,
              target_x_test: np.array,
              target_y_test: np.array,
              mitigate_size: int
            ):
    # Train the classifier
    assert len(target_x_train) > mitigate_size
    if mitigate_size == 0:
        x_train = source_x_train
        y_train = source_y_train
    mitigate_x_train = target_x_train[:mitigate_size]
    mitigate_y_train = target_y_train[:mitigate_size]
    x_train = np.concatenate([source_x_train, mitigate_x_train])
    y_train = np.concatenate([source_y_train, mitigate_y_train])
    if method == 'rank':
        # Standardize the data, for faster convergence
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        target_x_test = scaler.transform(target_x_test)
    clf = SVC(kernel='linear',probability=True, random_state=0).fit(x_train, y_train)
    train_result = run_clf(clf, x_train, y_train)
    test_result = run_clf(clf, target_x_test, target_y_test)
    return train_result, test_result


def find_threshold(method, train_scores, train_labels):
    # Sort scores to get possible threshold values
    thresholds = np.sort(train_scores)
    best_threshold = None
    best_f1 = 0
    if method in ['ll', 'fast-detectGPT',]:
        for t in thresholds:
            predictions = (train_scores > t).astype(int)
            f1 = f1_score(train_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

    elif method in ['rank', 'LRR', 'rank_GLTR', 'entropy', 'Binoculars' ]:
        for t in thresholds:
            predictions = (train_scores < t).astype(int)
            f1 = f1_score(train_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

    return best_threshold


def train_threshold(method: str,
                    source_x_train: np.array,
                    source_y_train: np.array,
                    target_x_train: np.array,
                    target_y_train: np.array,
                    target_x_test: np.array,
                    target_y_test: np.array,
                    mitigate_size: int
                    ):
    
    assert len(target_x_train) > mitigate_size
    if mitigate_size == 0:
        x_train = source_x_train
        y_train = source_y_train
    mitigate_x_train = target_x_train[:mitigate_size]
    mitigate_y_train = target_y_train[:mitigate_size]
    x_train = np.concatenate([source_x_train, mitigate_x_train])
    y_train = np.concatenate([source_y_train, mitigate_y_train])
    threshold = find_threshold(method, x_train, y_train)
    if method in ['rank', 'LRR', 'entropy', 'Binoculars']:
        y_train_preds = [x[0] < threshold for x in x_train]
        y_test_preds = [x[0] < threshold for x in target_x_test]
        train_result = y_train, y_train_preds, -1 * x_train # human has higher score
        test_result = target_y_test, y_test_preds, -1 * target_x_test

    elif method in ['ll', 'fast-detectGPT']:
        y_train_preds = [x[0] > threshold for x in x_train]
        y_test_preds = [x[0] > threshold for x in target_x_test]
        train_result = y_train, y_train_preds, x_train
        test_result = target_y_test, y_test_preds, target_x_test
    
    return train_result, test_result


def log_domain_result(csv_file: str, 
                      train_result: DetectOutput,
                      test_result: DetectOutput,
                      detectLLM: str, 
                      source_topic: str, 
                      target_topic: str, 
                      mitigate_size: int,
                      method: str,
                      criterion: str,
                      ):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['method', 'criterion', 'detect_llm', 'source_topic', 'target_topic', 'mitigate_size',
                                'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc',
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([method, criterion, detectLLM, source_topic, target_topic, mitigate_size,
                            round(train_result.acc, 4), round(train_result.precision, 4), round(train_result.recall, 4), round(train_result.f1, 4), round(train_result.auc, 4),
                            round(test_result.acc, 4), round(test_result.precision, 4), round(test_result.recall, 4), round(test_result.f1, 4), round(test_result.auc, 4)])


def log_llm_result( csv_file: str,
                    train_result: DetectOutput,
                    test_result: DetectOutput,
                    source_llm: str,
                    target_llm: str,
                    source_topic: str,
                    mitigate_size: int,
                    method: str,
                    criterion: str,
                    ):
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['method', 'criterion', 'source_llm', 'target_llm', 'source_topic', 'mitigate_size',
                                'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc',
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([method, criterion, source_llm, target_llm, source_topic, mitigate_size,
                            round(train_result.acc, 4), round(train_result.precision, 4), round(train_result.recall, 4), round(train_result.f1, 4), round(train_result.auc, 4),
                            round(test_result.acc, 4), round(test_result.precision, 4), round(test_result.recall, 4), round(test_result.f1, 4), round(test_result.auc, 4)])

# Binary, mitigation
def get_binary_domain_result():
    # change to yours
    domain_csv = './results/transfer_mitigate/zeroshot_mitigate_domain.csv'
    for method in ZEROSHOT_METHODS[:]:
        # domain transfer
        for detectLLM in LLMS:
            print(f'Processing {method} {detectLLM}')
            for source_topic in TOPICS:
                for target_topic in TOPICS:
                    if source_topic != target_topic:
                        sizes = [0, 100, 200, 300, 500, 800]
                    else:
                        sizes = [0]
                    # make sure you have the scores saved
                    source_x_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_{detectLLM}_x_train_{TRAIN_SIZE}.npy'
                    source_y_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_{detectLLM}_y_train_{TRAIN_SIZE}.npy'
                    target_x_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_{detectLLM}_x_train_{TRAIN_SIZE}.npy'
                    target_y_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_{detectLLM}_y_train_{TRAIN_SIZE}.npy'
                    target_x_test = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_{detectLLM}_x_test_{TEST_SIZE}.npy'
                    target_y_test = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_{detectLLM}_y_test_{TEST_SIZE}.npy'
                    source_x_train = np.load(source_x_train)
                    source_y_train = np.load(source_y_train)
                    target_x_train = np.load(target_x_train)
                    target_y_train = np.load(target_y_train)
                    target_x_test = np.load(target_x_test)
                    target_y_test = np.load(target_y_test)

                    for mitigate_size in sizes:
                        # logistic
                        train_result, test_result = train_logistic(source_x_train, source_y_train, target_x_train, target_y_train, target_x_test, target_y_test, mitigate_size)
                        train_metrics = cal_metrics(*train_result)
                        test_metrics = cal_metrics(*test_result)
                        log_domain_result(domain_csv, train_metrics, test_metrics, detectLLM, source_topic, target_topic, mitigate_size, method=method, criterion='logistic')
                        # threshold
                        if method == 'rank_GLTR':
                            continue
                        train_result, test_result = train_threshold(method, source_x_train, source_y_train, target_x_train, target_y_train, target_x_test, target_y_test, mitigate_size)
                        train_metrics = cal_metrics(*train_result)
                        test_metrics = cal_metrics(*test_result)
                        log_domain_result(domain_csv, train_metrics, test_metrics, detectLLM, source_topic, target_topic, mitigate_size, method=method, criterion='threshold')


def get_binary_llm_result():
    # change to yours
    llm_csv = './results/transfer_mitigate/zeroshot_mitigate_llm.csv'

    for method in ZEROSHOT_METHODS[:]:
        # llm transfer
        for source_topic in TOPICS:
            print(f'Processing {method} {source_topic}')
            for source_llm in LLMS:
                for target_llm in LLMS:
                    if source_llm != target_llm:
                        sizes = [0, 100, 200, 300, 500, 800]
                    else:
                        sizes = [0]
                    # make sure you have the scores saved
                    source_x_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_{source_llm}_x_train_{TRAIN_SIZE}.npy'
                    source_y_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_{source_llm}_y_train_{TRAIN_SIZE}.npy'
                    target_x_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_{target_llm}_x_train_{TRAIN_SIZE}.npy'
                    target_y_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_{target_llm}_y_train_{TRAIN_SIZE}.npy'
                    target_x_test = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_{target_llm}_x_test_{TEST_SIZE}.npy'
                    target_y_test = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_{target_llm}_y_test_{TEST_SIZE}.npy'
                    source_x_train = np.load(source_x_train)
                    source_y_train = np.load(source_y_train)
                    target_x_train = np.load(target_x_train)
                    target_y_train = np.load(target_y_train)
                    target_x_test = np.load(target_x_test)
                    target_y_test = np.load(target_y_test)

                    for mitigate_size in sizes:
                        # logistic
                        train_result, test_result = train_logistic(source_x_train, source_y_train, target_x_train, target_y_train, target_x_test, target_y_test, mitigate_size)
                        train_metrics = cal_metrics(*train_result)
                        test_metrics = cal_metrics(*test_result)
                        log_llm_result(llm_csv, train_metrics, test_metrics, source_llm, target_llm, source_topic, mitigate_size, method=method, criterion='logistic')
                        # threshold
                        if method == 'rank_GLTR':
                            continue
                        train_result, test_result = train_threshold(method, source_x_train, source_y_train, target_x_train, target_y_train, target_x_test, target_y_test, mitigate_size)
                        train_metrics = cal_metrics(*train_result)
                        test_metrics = cal_metrics(*test_result)
                        log_llm_result(llm_csv, train_metrics, test_metrics, source_llm, target_llm, source_topic, mitigate_size, method=method, criterion='threshold')


# Attribution, mitigation
def log_attribution_result(csv_file: str, 
                      train_result: DetectOutput,
                      test_result: DetectOutput,
                      source_topic: str, 
                      target_topic: str, 
                      mitigate_size: int,
                      method: str,
                      criterion: str,
                      ):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['method', 'criterion', 'source_topic', 'target_topic', 'mitigate_size',
                                'train_acc', 'train_precision', 'train_recall', 'train_f1', 'train_auc',
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc'])

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([method, criterion, source_topic, target_topic, mitigate_size,
                            round(train_result.acc, 4), round(train_result.precision, 4), round(train_result.recall, 4), round(train_result.f1, 4), round(train_result.auc, 4),
                            round(test_result.acc, 4), round(test_result.precision, 4), round(test_result.recall, 4), round(test_result.f1, 4), round(test_result.auc, 4)])


def get_attribution_result():
    # change to yours
    domain_csv = './results/transfer_mitigate/zeroshot_attribution_mitigate_domain.csv'
    for method in ZEROSHOT_METHODS[:]:
        # domain transfer
        for source_topic in TOPICS:
            print(f'Processing {method}, source topic: {source_topic}')
            for target_topic in TOPICS:
                if source_topic != target_topic:
                    sizes = [0, 100, 200, 300, 500, 800]
                else:
                    sizes = [0]
                # make sure you have the scores saved
                source_x_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_attribution_x_train_{TRAIN_SIZE}.npy'
                source_y_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{source_topic}_attribution_y_train_{TRAIN_SIZE}.npy'
                target_x_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_attribution_x_train_{TRAIN_SIZE}.npy'
                target_y_train = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_attribution_y_train_{TRAIN_SIZE}.npy'
                target_x_test = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_attribution_x_test_{TEST_SIZE}.npy'
                target_y_test = f'/data_sda/zhiyuan/zeroshot_scores/{method}_{target_topic}_attribution_y_test_{TEST_SIZE}.npy'
                source_x_train = np.load(source_x_train)
                source_y_train = np.load(source_y_train)
                target_x_train = np.load(target_x_train)
                target_y_train = np.load(target_y_train)
                target_x_test = np.load(target_x_test)
                target_y_test = np.load(target_y_test)

                for mitigate_size in sizes:
                    # logistic
                    train_result, test_result = train_logistic(source_x_train, source_y_train, target_x_train, target_y_train, target_x_test, target_y_test, mitigate_size)
                    train_metrics = cal_metrics(*train_result)
                    test_metrics = cal_metrics(*test_result)
                    log_attribution_result(domain_csv, train_metrics, test_metrics, source_topic, target_topic, mitigate_size, method=method, criterion='logistic')
                    # svm
                    train_result, test_result = train_svm(method, source_x_train, source_y_train, target_x_train, target_y_train, target_x_test, target_y_test, mitigate_size)
                    train_metrics = cal_metrics(*train_result)
                    test_metrics = cal_metrics(*test_result)
                    log_attribution_result(domain_csv, train_metrics, test_metrics, source_topic, target_topic, mitigate_size, method=method, criterion='svm')

if __name__ == '__main__':
    get_binary_domain_result()
    get_binary_llm_result()
    get_attribution_result()
