import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from sklearn import metrics
import pandas as pd

def get_output_target(test_csv, result_file):
    # groundtruth
    test_df = pd.read_csv(test_csv)
    gt_dict = {'G': 0, 'O': 1, 'A': 2}
    gt = [gt_dict[test_df.loc[idx, 'class']] for idx in range(len(test_df))]
    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.argmax(prob))
    return pred, gt

def avg_precision(test_csv, result_file):
    pred, gt = get_output_target(test_csv, result_file)
    # record multi class precision
    precision = metrics.precision_score(gt, pred, average=None)
    return precision

def avg_recall(test_csv, result_file):
    pred, gt = get_output_target(test_csv, result_file)
    # record multi class recall
    recall = metrics.recall_score(gt, pred, average=None)
    return recall

def confusion_matrix(test_csv, result_file):
    pred, gt = get_output_target(test_csv, result_file)
    return metrics.confusion_matrix(gt, pred, labels=[0,1])

def metrics_all(test_csv, result_file):
    labels = [0, 1, 2]
    pred, gt = get_output_target(test_csv, result_file)
    # metrics
    recall = metrics.recall_score(gt, pred, average=None)
    precision = metrics.precision_score(gt, pred, average=None)
    cm = metrics.confusion_matrix(gt, pred, labels=labels)
    f1 = metrics.f1_score(gt, pred, labels=labels, average='micro')
    #auroc = metrics.roc_auc_score(gt, pred, average='macro', multi_class='ovr', max_fpr=1.0)
    return precision, recall, f1, cm
