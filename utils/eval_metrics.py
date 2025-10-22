"""
Evaluation metrics for EGA-Ploc protein subcellular localization / EGA-Ploc蛋白质亚细胞定位的评估指标
This module contains comprehensive evaluation metrics for multi-label classification tasks.
此模块包含多标签分类任务的全面评估指标。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.tensorboard import SummaryWriter

from models.criterion import t_criterion, max_criterion
import utils.distributed as du
from utils.config_defaults import labelLists


np.set_printoptions(suppress=True)

epsilon = 1e-9


def compute_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall + epsilon)


def _example_based_quantity(labels, preds):
    ex_equal = np.all(np.equal(labels, preds), axis=1).astype("float32")
    ex_and = np.sum(np.logical_and(labels, preds), axis=1).astype("float32")
    ex_or = np.sum(np.logical_or(labels, preds), axis=1).astype("float32")
    ex_predict = np.sum(preds, axis=1).astype("float32")
    ex_ground_truth = np.sum(labels, axis=1).astype("float32")
    ex_xor = np.logical_xor(labels, preds).astype("float32")

    return ex_equal, ex_and, ex_or, ex_predict, ex_ground_truth, ex_xor


def example_metrics(ex_equal, ex_and, ex_or, ex_predict, ex_ground_truth, ex_xor):
    subset_accuracy = np.mean(ex_equal)
    accuracy = np.mean((ex_and + epsilon) / (ex_or + epsilon))
    precision = np.mean((ex_and + epsilon) / (ex_predict + epsilon))
    recall = np.mean((ex_and + epsilon) / (ex_ground_truth + epsilon))
    f1 = compute_f1(precision, recall)
    hamming_loss = np.mean(ex_xor)

    return subset_accuracy, accuracy, precision, recall, f1, hamming_loss


def _label_quantity(labels, preds):
    tp = np.sum(np.logical_and(labels, preds), axis=0)
    fp = np.sum(np.logical_and(1 - labels, preds), axis=0)
    tn = np.sum(np.logical_and(1 - labels, 1 - preds), axis=0)
    fn = np.sum(np.logical_and(labels, 1 - preds), axis=0)
    return np.stack([tp, fp, tn, fn], axis=0).astype("float32")


def cal_label_metrics(tp, fp, tn, fn):
    accuracy = np.mean((tp + tn + epsilon) / (tp + fp + tn + fn + epsilon))
    precision = np.mean((tp + epsilon) / (tp + fp + epsilon))
    recall = np.mean((tp + epsilon) / (tp + fn + epsilon))
    f1 = compute_f1(precision, recall)
    jaccard = np.mean((tp + epsilon) / (tp + fp + fn + epsilon))

    return accuracy, precision, recall, f1, jaccard


def label_macro_metrics(quantity):
    tp, fp, tn, fn = quantity

    return cal_label_metrics(tp, fp, tn, fn)


def label_micro_metrics(quantity):
    tp, fp, tn, fn = np.sum(quantity, axis=1)

    return cal_label_metrics(tp, fp, tn, fn)


def every_label_metrics(quantity):
    tp, fp, tn, fn = quantity

    accuracy = (tp + tn + epsilon) / (tp + fp + tn + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    f1 = compute_f1(precision, recall)
    jaccard = (tp + epsilon) / (tp + fp + fn + epsilon)

    return accuracy, precision, recall, f1, jaccard


def SPE_metrics(quantity):
    tp, fp, tn, fn = quantity
    label_SPE_macro = np.mean((tn + epsilon) / (tn + fp + epsilon))

    tp, fp, tn, fn = np.sum(quantity, axis=1)
    label_SPE_micro = (tn + epsilon) / (tn + fp + epsilon)

    return label_SPE_macro, label_SPE_micro


def auc_metrics(labels, preds, locations):
    fpr, tpr, thres = roc_curve(labels.ravel(), preds.ravel())
    micro_auc = auc(fpr, tpr)

    auc_list = []
    for i in range(len(locations)):
        fpr, tpr, thres = roc_curve(labels[:, i], preds[:, i])
        if (not np.isnan(tpr).any()) and (not np.isnan(fpr).any()):
            auc_list.append(auc(fpr, tpr))
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)

    return micro_auc, mean_auc, std_auc


def mcc_metrics(labels, preds, locations):
    G = len(locations)
    confusion_matrix = np.zeros([G, G])
    for i in range(G):
        for j in range(G):
            confusion_matrix[i, j] = np.sum(np.logical_and(labels[:, j], preds[:, i]))

    top = 0
    for g in range(G):
        for j in range(G):
            for r in range(G):
                top += (confusion_matrix[g, g] * confusion_matrix[j, r] - confusion_matrix[g, j] * confusion_matrix[r, g])
    bottom1 = 0
    for g in range(G):
        sum1 = np.sum(confusion_matrix[g, :])
        sum2 = 0
        for g1 in range(G):
            if g == g1:
                continue
            sum2 += np.sum(confusion_matrix[g1, :])
        bottom1 += sum1 * sum2
    bottom2 = 0
    for g in range(G):
        sum1 = np.sum(confusion_matrix[:, g])
        sum2 = 0
        for g2 in range(G):
            if g == g2:
                continue
            sum2 += np.sum(confusion_matrix[:, g2])
        bottom2 += sum1 * sum2
    bottom = np.sqrt(bottom1) * np.sqrt(bottom2)

    mcc = top / bottom
    return mcc


def cal_metrics(cfg, labels, preds, writer=None, cur_epoch=None, locations=None, csvWriter=None, randomSplit=-1, fold=0, thresh='', split='', getQuantity=False, getSPE=False, getAuc=False, getMcc=False):
    ex_equal, ex_and, ex_or, ex_predict, ex_ground_truth, ex_xor = _example_based_quantity(labels, preds)
    quantity = _label_quantity(labels, preds)

    ex_subset_acc, ex_acc, ex_precision, ex_recall, ex_f1, ex_hamming_loss = example_metrics(ex_equal, ex_and, ex_or, ex_predict, ex_ground_truth, ex_xor)
    lab_acc_macro, lab_precision_macro, lab_recall_macro, lab_f1_macro, lab_jaccard_macro = label_macro_metrics(quantity)
    lab_acc_micro, lab_precision_micro, lab_recall_micro, lab_f1_micro, lab_jaccard_micro = label_micro_metrics(quantity)

    lab_acc, lab_precision, lab_recall, lab_f1, lab_jaccard = every_label_metrics(quantity)

    print("example_subset_accuracy:", ex_subset_acc)
    print("example_accuracy:", ex_acc)
    print("example_precision:", ex_precision)
    print("example_recall:", ex_recall)
    print("example_f1:", ex_f1)
    print("example_hamming_loss:", ex_hamming_loss)
    print()

    print("label_accuracy_macro:", lab_acc_macro)
    print("label_precision_macro:", lab_precision_macro)
    print("label_recall_macro:", lab_recall_macro)
    print("label_f1_macro:", lab_f1_macro)
    print("label_jaccard_macro:", lab_jaccard_macro)
    print()


    print("label_accuracy_micro:", lab_acc_micro)
    print("label_precision_micro:", lab_precision_micro)
    print("label_recall_micro:", lab_recall_micro)
    print("label_f1_micro:", lab_f1_micro)
    print("label_jaccard_micro:", lab_jaccard_micro)
    print()

    if getSPE:
        label_SPE_macro, label_SPE_micro = SPE_metrics(quantity)
        print("label_specificity_macro:", label_SPE_macro)
        print("label_specificity_micro:", label_SPE_micro)
        print()
    if getAuc:
        micro_auc, mean_auc, std_auc = auc_metrics(labels, preds, locations)
        print("auc:", micro_auc)
        print("mean_auc:", mean_auc)
        print("std_auc:", std_auc)
        print()
    if getMcc:
        mcc = mcc_metrics(labels, preds, locations)


    if locations == None:
        locations = cfg.CLASSIFIER.LOCATIONS
    print("label:", locations)
    print("label_accuracy:", lab_acc)
    print("label_precision:", lab_precision)
    print("label_recall:", lab_recall)
    print("label_f1:", lab_f1)
    print("label_jaccard:", lab_jaccard)
    print()

    if writer:
        writer.add_scalar(tag="example/subset_accuracy", scalar_value=ex_subset_acc, global_step=cur_epoch)
        writer.add_scalar(tag="example/accuracy", scalar_value=ex_acc, global_step=cur_epoch)
        writer.add_scalar(tag="example/precision", scalar_value=ex_precision, global_step=cur_epoch)
        writer.add_scalar(tag="example/recall", scalar_value=ex_recall, global_step=cur_epoch)
        writer.add_scalar(tag="example/f1", scalar_value=ex_f1, global_step=cur_epoch)
        writer.add_scalar(tag="example/hamming_loss", scalar_value=ex_hamming_loss, global_step=cur_epoch)

        writer.add_scalar(tag="label_macro/accuracy", scalar_value=lab_acc_macro, global_step=cur_epoch)
        writer.add_scalar(tag="label_macro/precision", scalar_value=lab_precision_macro, global_step=cur_epoch)
        writer.add_scalar(tag="label_macro/recall", scalar_value=lab_recall_macro, global_step=cur_epoch)
        writer.add_scalar(tag="label_macro/f1", scalar_value=lab_f1_macro, global_step=cur_epoch)
        writer.add_scalar(tag="label_macro/jaccard", scalar_value=lab_jaccard_macro, global_step=cur_epoch)

        writer.add_scalar(tag="label_micro/accuracy", scalar_value=lab_acc_micro, global_step=cur_epoch)
        writer.add_scalar(tag="label_micro/precision", scalar_value=lab_precision_micro, global_step=cur_epoch)
        writer.add_scalar(tag="label_micro/recall", scalar_value=lab_recall_micro, global_step=cur_epoch)
        writer.add_scalar(tag="label_micro/f1", scalar_value=lab_f1_micro, global_step=cur_epoch)
        writer.add_scalar(tag="label_micro/jaccard", scalar_value=lab_jaccard_micro, global_step=cur_epoch)

        for i in range(len(locations)):
            writer.add_scalar(tag="{}/accuracy".format(locations[i]), scalar_value=lab_acc[i], global_step=cur_epoch)
            writer.add_scalar(tag="{}/precision".format(locations[i]), scalar_value=lab_precision[i], global_step=cur_epoch)
            writer.add_scalar(tag="{}/recall".format(locations[i]), scalar_value=lab_recall[i], global_step=cur_epoch)
            writer.add_scalar(tag="{}/f1".format(locations[i]), scalar_value=lab_f1[i], global_step=cur_epoch)
            writer.add_scalar(tag="{}/jaccard".format(locations[i]), scalar_value=lab_jaccard[i], global_step=cur_epoch)

    if csvWriter:
        result_row = [randomSplit, fold, thresh, split]
        if getQuantity:
            result_row.extend(np.sum(quantity, axis=1).tolist())
            result_row.extend(quantity.flatten('f'))
        result_row.extend([ex_subset_acc, ex_acc, ex_precision, ex_recall, ex_f1, ex_hamming_loss])
        result_row.extend([lab_acc_macro, lab_precision_macro, lab_recall_macro, lab_f1_macro, lab_jaccard_macro])
        result_row.extend([lab_acc_micro, lab_precision_micro, lab_recall_micro, lab_f1_micro, lab_jaccard_micro])
        if getSPE:
            result_row.extend([label_SPE_macro, label_SPE_micro])
        if getAuc:
            result_row.extend([micro_auc, mean_auc, std_auc])
        if getMcc:
            result_row.extend([mcc])
        for i in range(len(locations)):
            result_row.extend([lab_acc[i], lab_precision[i], lab_recall[i], lab_f1[i], lab_jaccard[i]])
        print(result_row)
        csvWriter.writerow(result_row)

    return quantity

def test_cal_metrics(cfg, labels, preds, locations=None):
    ex_equal, ex_and, ex_or, ex_predict, ex_ground_truth, ex_xor = _example_based_quantity(labels, preds)
    quantity = _label_quantity(labels, preds)

    ex_subset_acc, ex_acc, ex_precision, ex_recall, ex_f1, ex_hamming_loss = example_metrics(ex_equal, ex_and, ex_or,
                                                                                             ex_predict,
                                                                                             ex_ground_truth, ex_xor)
    lab_acc_macro, lab_precision_macro, lab_recall_macro, lab_f1_macro, lab_jaccard_macro = label_macro_metrics(
        quantity)
    lab_acc_micro, lab_precision_micro, lab_recall_micro, lab_f1_micro, lab_jaccard_micro = label_micro_metrics(
        quantity)

    lab_acc, lab_precision, lab_recall, lab_f1, lab_jaccard = every_label_metrics(quantity)

    print("example_subset_accuracy:", ex_subset_acc)
    print("example_accuracy:", ex_acc)
    print("example_precision:", ex_precision)
    print("example_recall:", ex_recall)
    print("example_f1:", ex_f1)
    print("example_hamming_loss:", ex_hamming_loss)
    print()

    print("label_accuracy_macro:", lab_acc_macro)
    print("label_precision_macro:", lab_precision_macro)
    print("label_recall_macro:", lab_recall_macro)
    print("label_f1_macro:", lab_f1_macro)
    print("label_jaccard_macro:", lab_jaccard_macro)
    print()

    print("label_accuracy_micro:", lab_acc_micro)
    print("label_precision_micro:", lab_precision_micro)
    print("label_recall_micro:", lab_recall_micro)
    print("label_f1_micro:", lab_f1_micro)
    print("label_jaccard_micro:", lab_jaccard_micro)
    print()

    if locations == None:
        locations = cfg.CLASSIFIER.LOCATIONS
    print("label:", locations)
    print("label_accuracy:", lab_acc)
    print("label_precision:", lab_precision)
    print("label_recall:", lab_recall)
    print("label_f1:", lab_f1)
    print("label_jaccard:", lab_jaccard)
    print()


def get_curve(cfg, labels, preds, optimal_func="f_beta", beta=[0.5 for i in range(10)], writer=None, locations=None):
    if locations == None:
        locations = cfg.CLASSIFIER.LOCATIONS
    optimal_thres = []
    for i in range(len(locations)):
        fpr, tpr, thres = roc_curve(labels[:, i], preds[:, i])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="AUC: {:.2f}".format(auc(fpr, tpr)))
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(locations[i])
        ax.legend(loc="lower right")

        if writer:
            writer.add_figure(tag="roc_curve/{}".format(locations[i]), figure=fig, global_step=1)

        if beta[i] == -1:
            optimal_thres.append(0.5)
        else:
            precision, recall, thresholds = precision_recall_curve(labels[:, i], preds[:, i])
            f_beta = ((1 + beta[i] ** 2) * precision * recall) / (beta[i] ** 2 * precision + recall + epsilon)
            f_beta_idx = np.argmax(f_beta)
            optimal_threshold = thresholds[f_beta_idx]
            optimal_thres.append(optimal_threshold)

        if writer:
            writer.add_pr_curve("pr_curve/{}".format(locations[i]), labels[:, i], preds[:, i], 0)
    print()

    return optimal_thres


def evaluate(cfg, all_idxs, all_labels, all_preds, data_file, model_name, result_prefix=None, log_prefix=None, metricsWriter=None, cur_epoch=-1,
        get_threshold=False, threshold=0.5, multilabel=True, prefix="", beta=[0.5 for i in range(10)], csvWriter=None, randomSplit=-1, fold=0, aug=0, thresh='', split='', getSPE=False, getAuc=False, getMcc=False):
    locations = cfg.CLASSIFIER.LOCATIONS
    labels = labelLists.get(log_prefix.rsplit("/", 2)[0], locations)
    locations_pred = [i + '_pred' for i in labels]
    locations_pred_labels = [i + '_pred_labels' for i in labels]

    all_labels = pd.DataFrame(all_labels, columns=locations)
    all_preds = pd.DataFrame(all_preds, columns=locations)
    all_labels = np.array(all_labels[labels])
    all_preds = np.array(all_preds[labels])

    if get_threshold:
        writer = SummaryWriter(log_dir="logs/{}/curve/{}".format(log_prefix, data_file.split('/')[-1]))
        threshold = get_curve(cfg, all_labels, all_preds, beta=beta, writer=writer, locations=labels)
        writer.close()

    if multilabel:
        all_pred_labels = t_criterion(all_preds, threshold or 0.5)
    else:
        all_pred_labels = max_criterion(all_preds)
    cal_metrics(cfg, all_labels, all_pred_labels, metricsWriter, cur_epoch, locations=labels, csvWriter=csvWriter, randomSplit=randomSplit, fold=fold, thresh=thresh, split=split, getSPE=getSPE, getAuc=getAuc, getMcc=getMcc)

    labeledData = pd.read_csv(data_file, header=0, index_col=0)
    predData = pd.DataFrame(columns=locations_pred, index=all_idxs)
    predData[locations_pred] = all_preds
    predData[locations_pred_labels] = all_pred_labels
    predData = pd.merge(labeledData, predData, how='left', left_index=True, right_index=True)

    print("{}/{}/preds".format(result_prefix, model_name))
    if not os.path.exists("{}/{}/preds".format(result_prefix, model_name)):
        os.makedirs("{}/{}/preds".format(result_prefix, model_name))
    if cur_epoch != -1:
        predData.to_csv("{}/{}/preds/{}{}_{}".format(result_prefix, model_name, prefix, (cur_epoch + 1), data_file.split('/')[-1]), index=True, mode='w')
    predData.to_csv("{}/{}/preds/{}test_{}_aug{}_{}".format(result_prefix, model_name, prefix, thresh, aug, data_file.split('/')[-1]), index=True, mode='w')

    return threshold

def test_evaluate(cfg,all_labels, all_preds, log_prefix):
    locations = cfg.CLASSIFIER.LOCATIONS
    labels = labelLists.get(log_prefix.rsplit("/", 2)[0], locations)

    all_labels = pd.DataFrame(all_labels, columns=locations)
    all_preds = pd.DataFrame(all_preds, columns=locations)
    all_labels = np.array(all_labels[labels])
    all_preds = np.array(all_preds[labels])

    all_pred_labels = t_criterion(all_preds, 0.5)
    test_cal_metrics(cfg, all_labels, all_pred_labels, locations=labels)
