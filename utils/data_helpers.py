# -*- coding:utf-8 -*-
__author__ = 'Randolph'
import torch
import os, sys
import time
import heapq
import gensim
import logging
import json
import numpy as np
from collections import OrderedDict
import math
from texttable import Texttable
from gensim.models import KeyedVectors
from skimage import io, transform, color
import torch
from torch.nn.functional import one_hot
from solt.core import DataContainer
sys.path.append('../')
from utils import xtree_utils as xtree 
from torchmetrics import AUROC, AveragePrecision
def _option(pattern):
    """
    Get the option according to the pattern.
    pattern 0: Choose training or restore.
    pattern 1: Choose best or latest checkpoint.

    Args:
        pattern: 0 for training step. 1 for testing step.
    Returns:
        The OPTION.
    """
    if pattern == 0:
        #OPTION = input("[Input] Train or Restore? (T/R): ")
        OPTION = 'T'
        while not (OPTION.upper() in ['T', 'R']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
            
    if pattern == 1:
        OPTION = input("Load Best or Latest Model? (B/L): ")
        while not (OPTION.isalpha() and OPTION.upper() in ['B', 'L']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger.
        input_file: The logger file path.
        level: The logger level.
    Returns:
        The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(input_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger



def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_out_dir(option, logger):
    """
    Get the out dir for saving model checkpoints.

    Args:
        option: Train or Restore.
        logger: The logger.
    Returns:
        The output dir for model checkpoints.
    """
    if option == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {0}\n".format(out_dir))
    if option == 'R':
        MODEL = input("[Input] Please input the checkpoints model you want to restore, "
                      "it should be like (1490175368): ")  # The model you want to restore

        while not (MODEL.isdigit() and len(MODEL) == 10):
            MODEL = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
        logger.info("Writing to {0}\n".format(out_dir))
    return out_dir


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name.
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(output_file, data_id, true_labels, predict_labels, predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network.
        data_id: The data record id info provided by dict <Data>.
        true_labels: The all true labels.
        predict_labels: The all predict labels by threshold.
        predict_scores: The all predict scores by threshold.
    Raises:
        IOError: If the prediction file is not a .json file.
    """
    if not output_file.endswith('.json'):
        raise IOError("[Error] The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(predict_labels)
        for i in range(data_size):
            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', [int(i) for i in true_labels[i]]),
                ('predict_labels', [int(i) for i in predict_labels[i]]),
                ('predict_scores', [round(i, 4) for i in predict_scores[i]])
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')
def generate_hierarchy_matrix_from_tree(hierarchy_tree):
    hierarchy_dicts = xtree.generate_dicts_per_level(hierarchy_tree)
    total_hierarchy_dict =  {}
    counter = 0 
    for hierarchy_dict in hierarchy_dicts:
        for key in hierarchy_dict.keys():
            total_hierarchy_dict[key] = counter
            counter+=1   

    hierarchy_matrix = np.zeros((len(total_hierarchy_dict),len(total_hierarchy_dict)))
    for key_parent,value_parent in total_hierarchy_dict.items():
        for key_child,value_child in total_hierarchy_dict.items():
            if key_parent == key_child:
                hierarchy_matrix[total_hierarchy_dict[key_parent],total_hierarchy_dict[key_parent]] = 1
            elif key_child.startswith(key_parent):
                hierarchy_matrix[total_hierarchy_dict[key_parent],total_hierarchy_dict[key_child]] = 1
    
    return hierarchy_matrix   


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted one-hot labels based on the threshold.

    Args:
        scores: The all classes predicted scores provided by network.
        threshold: The threshold (default: 0.5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels

#def get_onehot_label_threshold(scores, threshold=0.5):
#    """
#    Get the predicted one-hot labels based on the threshold.
#    If there is no predict score greater than threshold, then choose the label which has the max predict score.
#
#    Args:
#        scores: The all classes predicted scores provided by network.
#        threshold: The threshold (default: 0.5).
#    Returns:
#        predicted_onehot_labels: The predicted labels (one-hot).
#    """
#    predicted_onehot_labels = []
#    scores = np.ndarray.tolist(scores)
#    for score in scores:
#        count = 0
#        onehot_labels_list = [0] * len(score)
#        for index, predict_score in enumerate(score):
#            if predict_score >= threshold:
#                onehot_labels_list[index] = 1
#                count += 1
#        if count == 0:
#            max_score_index = score.index(max(score))
#            onehot_labels_list[max_score_index] = 1
#        predicted_onehot_labels.append(onehot_labels_list)
#    return predicted_onehot_labels

def get_pcp_onehot_label_threshold(scores,explicit_hierarchy,num_classes_list, pcp_threshold=-1.0):
    """
    Get the predicted one-hot labels based on PCP.

    Args:
        scores: The all classes predicted scores provided by network.
        explicit_hierarchy: The explicit hierarchy matrix.
        num_classes_list: List of Classes per Hierarchy Layer.
        pcp_threshold: The PCP-threshold (default: -1.0).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
    """
    path_pruned_classes_list = []
    for score in scores:
        path_pruned_classes = prune_based_coherent_prediction(score=score.tolist(),explicit_hierarchy=explicit_hierarchy,pcp_threshold=pcp_threshold,num_classes_list=num_classes_list)
        
        path_pruned_classes_list.append(path_pruned_classes)
    num_classes=scores.shape[1]
    pcp_onehot_labels = [torch.zeros(num_classes) if len(path_pruned_classes) == 0 else generate_one_hot(path_pruned_classes,num_classes) for path_pruned_classes in path_pruned_classes_list]
    return pcp_onehot_labels

def get_pcp_onehot_label_topk(scores,explicit_hierarchy,num_classes_list,pcp_threshold=-1.0, top_num=1):
    """
    Get the predicted one-hot labels based on PCP and topK.

    Args:
        scores: The all classes predicted scores provided by network.
        explicit_hierarchy: The explicit hierarchy matrix.
        num_classes_list: List of Classes per Hierarchy Layer.
        pcp_threshold: The PCP-threshold (default: -1.0).
        top_num: The max topK number (default: 5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
    """
    path_pruned_classes_topk_list = []
    num_classes = scores.shape[1]
    for score in scores:
        path_pruned_classes = prune_based_coherent_prediction(score=score.tolist(),explicit_hierarchy=explicit_hierarchy,num_classes_list=num_classes_list,pcp_threshold=pcp_threshold)
        pruned_score = torch.zeros(num_classes)
        for class_idx in path_pruned_classes:
            pruned_score[class_idx] = score[class_idx]
        #for class_idx in path_pruned_classes:
        #    pruned_score[class_idx] = score[class_idx]
        _,topk_index = torch.topk(pruned_score,k=top_num)
        path_pruned_classes_topk_list.append(topk_index)
    pcp_onehot_labels_topk = [torch.zeros(num_classes) if len(path_pruned_classes_topk) == 0 else generate_one_hot(path_pruned_classes_topk,num_classes) for path_pruned_classes_topk in path_pruned_classes_topk_list]
    return pcp_onehot_labels_topk      

def calc_metrics(scores_list,labels_list,topK,pcp_hierarchy,pcp_threshold,num_classes_list,device=None):
    metric_dict = {}
    
    eval_pre_pcp_tk = [0.0] * topK
    eval_rec_pcp_tk = [0.0] * topK
    eval_F1_pcp_tk = [0.0] * topK
    eval_emr_pcp_tk = [0.0] * topK
    eval_pre_tk = [0.0] * topK
    eval_rec_tk = [0.0] * topK
    eval_F1_tk = [0.0] * topK
    eval_emr_tk = [0.0] * topK
    predicted_onehot_labels_ts = []
    predicted_pcp_onehot_labels_ts = []
    predicted_onehot_labels_tk = [[] for _ in range(topK)]
    predicted_pcp_onehot_labels_tk = [[] for _ in range(topK)]
    scores = torch.cat([torch.unsqueeze(tensor,0) for tensor in scores_list],dim=0).to('cpu')
    scores_np = scores.numpy()
    # Predict by threshold
    batch_predicted_onehot_labels = get_onehot_label_threshold(scores=scores_np,threshold=0.5)
    for k in batch_predicted_onehot_labels:
        predicted_onehot_labels_ts.append(k)
    
    # Predict by topK
    for top_num in range(topK):
        batch_predicted_onehot_labels_tk = get_onehot_label_topk(scores=scores_np,top_num=top_num+1)
        for i in batch_predicted_onehot_labels_tk:
            predicted_onehot_labels_tk[top_num].append(i)
    
    # Predict by pcp-threshold
    batch_predicted_pcp_onehot_labels_ts = get_pcp_onehot_label_threshold(scores=scores,explicit_hierarchy=pcp_hierarchy,num_classes_list=num_classes_list, pcp_threshold=pcp_threshold)
    for k in batch_predicted_pcp_onehot_labels_ts:
        predicted_pcp_onehot_labels_ts.append(k)
    
    # Predict by pcp-topK
    for top_num in range(topK):
        batch_predicted_pcp_onehot_labels_tk = get_pcp_onehot_label_topk(scores=scores,explicit_hierarchy=pcp_hierarchy,pcp_threshold=pcp_threshold,num_classes_list=num_classes_list, top_num=top_num+1)
        for i in batch_predicted_pcp_onehot_labels_tk:
            predicted_pcp_onehot_labels_tk[top_num].append(i)
            
    # Calculate PCP Precision & Recall & F1
    true_onehot_labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in labels_list],dim=0).to(device)
    predicted_pcp_onehot_labels = torch.cat([torch.unsqueeze(tensor,0) for tensor in predicted_pcp_onehot_labels_ts],dim=0).to(device)
    
    eval_pre_pcp_ts,eval_rec_pcp_ts,eval_F1_pcp_ts = precision_recall_f1_score(labels=true_onehot_labels,binary_predictions=predicted_pcp_onehot_labels, average='micro')
    
    for top_num in range(topK):
        predicted_pcp_onehot_labels_topk = torch.cat([torch.unsqueeze(tensor,0) for tensor in predicted_pcp_onehot_labels_tk[top_num]],dim=0).to(device)
        eval_pre_pcp_tk[top_num], eval_rec_pcp_tk[top_num],eval_F1_pcp_tk[top_num] = precision_recall_f1_score(labels=true_onehot_labels,binary_predictions=predicted_pcp_onehot_labels_topk, average='micro')
        eval_emr_pcp_tk[top_num] = eval_exact_match_ratio(true_labels_batch=true_onehot_labels,predicted_labels_batch=predicted_pcp_onehot_labels_topk)
    # Calculate Precision & Recall & F1
    predicted_onehot_labels = torch.cat([torch.unsqueeze(torch.tensor(tensor),0) for tensor in predicted_onehot_labels_ts],dim=0).to(device)
    
    eval_pre_ts,eval_rec_ts,eval_F1_ts = precision_recall_f1_score(labels=true_onehot_labels,binary_predictions=predicted_onehot_labels, average='micro')
        
    for top_num in range(topK):
        predicted_onehot_labels_topk = torch.cat([torch.unsqueeze(torch.tensor(tensor),0) for tensor in predicted_onehot_labels_tk[top_num]],dim=0).to(device)
        eval_pre_tk[top_num], eval_rec_tk[top_num],eval_F1_tk[top_num] = precision_recall_f1_score(labels=true_onehot_labels,binary_predictions=predicted_onehot_labels_topk, average='micro')
        eval_emr_tk[top_num] = eval_exact_match_ratio(true_labels_batch=true_onehot_labels,predicted_labels_batch=predicted_onehot_labels_topk)
    
    # Calculate Exact Match-Ratio
    eval_emr_ts = eval_exact_match_ratio(true_labels_batch=true_onehot_labels,predicted_labels_batch=predicted_onehot_labels)
    
    # Calculate Exact Match-Ratio for PCP Threshold
    eval_emr_pcp_ts = eval_exact_match_ratio(true_labels_batch=true_onehot_labels,predicted_labels_batch=predicted_pcp_onehot_labels)
    
    # Calculate Precision & Recall & F1 per Hierarchy-Layer
    eval_metrics_per_layer = get_per_layer_metrics(scores=predicted_onehot_labels,labels=true_onehot_labels,num_classes_list=num_classes_list)
    
    # Calculate PCP Precision & Recall & F1 per Hierarchy-Layer
    eval_pcp_metrics_per_layer = get_per_layer_metrics(scores=predicted_pcp_onehot_labels,labels=true_onehot_labels,num_classes_list=num_classes_list)
    
    
            
    auroc = AUROC(task="binary")
    eval_pcp_auc = auroc(predicted_pcp_onehot_labels,true_onehot_labels.to(dtype=torch.long))
    
    auprc = AveragePrecision(task="binary")
    eval_pcp_auprc = auprc(predicted_pcp_onehot_labels,true_onehot_labels.to(dtype=torch.long))
    print(predicted_onehot_labels.to(dtype=torch.float32)[:10])
    eval_auc = auroc(predicted_onehot_labels.to(dtype=torch.float32),true_onehot_labels.to(dtype=torch.long))
    eval_auprc = auprc(predicted_onehot_labels.to(dtype=torch.float32),true_onehot_labels.to(dtype=torch.long))
    metric_dict['Validation/PCPAverageAUC'] = eval_pcp_auc
    metric_dict['Validation/PCPAveragePrecision'] = eval_pcp_auprc
    metric_dict['Validation/PCPPrecision'] = eval_pre_pcp_ts
    metric_dict['Validation/PCPRecall'] = eval_rec_pcp_ts
    metric_dict['Validation/PCPF1'] = eval_F1_pcp_ts
    metric_dict['Validation/PCPEMR'] = eval_emr_pcp_ts
    for i,pcp_precision in enumerate(eval_pre_pcp_tk):
        metric_dict[f'Validation/PCPPrecisionTopK/{i}'] = pcp_precision
    for i, pcp_recall in enumerate(eval_rec_pcp_tk):
        metric_dict[f'Validation/PCPRecallTopK/{i}'] = pcp_recall
    for i, pcp_f1 in enumerate(eval_F1_pcp_tk):
        metric_dict[f'Validation/PCPF1TopK/{i}'] = pcp_f1
    for i, pcp_emr in enumerate(eval_emr_pcp_tk):
        metric_dict[f'Validation/PCPEMRTopK/{i}'] = pcp_emr
    
    metric_dict['Validation/AverageAUC'] = eval_auc
    metric_dict['Validation/AveragePrecision'] = eval_auprc
    metric_dict['Validation/Precision'] = eval_pre_ts
    metric_dict['Validation/Recall'] = eval_rec_ts
    metric_dict['Validation/F1'] = eval_F1_ts
    metric_dict['Validation/EMR'] = eval_emr_ts

    for i, precision in enumerate(eval_pre_tk):
        metric_dict[f'Validation/PrecisionTopK/{i}'] = precision
    for i, recall in enumerate(eval_rec_tk):
        metric_dict[f'Validation/RecallTopK/{i}'] = recall
    for i, f1 in enumerate(eval_F1_tk):
        metric_dict[f'Validation/F1TopK/{i}'] = f1
    for i, emr in enumerate(eval_emr_tk):
        metric_dict[f'Validation/EMRTopK/{i}'] = emr
    
    for i in range(len(eval_metrics_per_layer)):
        eval_layer_pre = eval_metrics_per_layer[i]['pre']
        eval_layer_rec = eval_metrics_per_layer[i]['rec']
        eval_layer_f1 = eval_metrics_per_layer[i]['f1']
        eval_layer_auc = eval_metrics_per_layer[i]['auc']
        eval_layer_auprc = eval_metrics_per_layer[i]['auprc']
        eval_layer_pcp_pre = eval_pcp_metrics_per_layer[i]['pre']
        eval_layer_pcp_rec = eval_pcp_metrics_per_layer[i]['rec']
        eval_layer_pcp_f1 = eval_pcp_metrics_per_layer[i]['f1']
        eval_layer_pcp_auc = eval_pcp_metrics_per_layer[i]['auc']
        eval_layer_pcp_auprc = eval_pcp_metrics_per_layer[i]['auprc']
        metric_dict[f'Validation/{i+1}-LayerPrecision'] = eval_layer_pre
        metric_dict[f'Validation/{i+1}-LayerRecall'] = eval_layer_rec
        metric_dict[f'Validation/{i+1}-LayerF1'] = eval_layer_f1
        metric_dict[f'Validation/{i+1}-LayerAUC'] = eval_layer_auc
        metric_dict[f'Validation/{i+1}-LayerAUPRC'] = eval_layer_auprc
        metric_dict[f'Validation/{i+1}-LayerPCPPrecision'] = eval_layer_pcp_pre
        metric_dict[f'Validation/{i+1}-LayerPCPRecall'] = eval_layer_pcp_rec
        metric_dict[f'Validation/{i+1}-LayerPCPF1'] = eval_layer_pcp_f1
        metric_dict[f'Validation/{i+1}-LayerPCPAUC'] = eval_layer_pcp_auc
        metric_dict[f'Validation/{i+1}-LayerPCPAUPRC'] = eval_layer_pcp_auprc
    
    # Show metrics
    
    # Predict by threshold
    print("Predict by thresholding: Precision {0:g}, Recall {1:g}, F1 {2:g}, AUC {3:g} , AUPRC {4:g}, EMR {5:g}".format(eval_pre_ts, eval_rec_ts, eval_F1_ts,eval_auc,eval_auprc,eval_emr_ts))
    print("\n")
    # Predict by topK
    print("Predict by topK:")
    for top_num in range(topK):
        print("Top{0}: Precision {1:g}, Recall {2:g}, F1 {3:g}, EMR {4:g}".format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F1_tk[top_num],eval_emr_tk[top_num]))  
    print("\n")
    # Predict by threshold per layer
    print("Thresholding Prediction by Layer:")
    for i in range(len(eval_metrics_per_layer)):
        print("Layer{0}: Precision {1:g}, Recall {2:g}, F1 {3:g}, AUC {4:g}, AUPRC {5:g}, EMR {6:g}".format(i+1, eval_metrics_per_layer[i]['pre'], eval_metrics_per_layer[i]['rec'], eval_metrics_per_layer[i]['f1'],eval_metrics_per_layer[i]['auc'],eval_metrics_per_layer[i]['auprc'],eval_metrics_per_layer[i]['emr']))  
    print("\n")
    # Predict by pcp
    print("Predict by PCP thresholding: PCP-Precision {0:g}, PCP-Recall {1:g}, PCP-F1 {2:g}, PCP-AUC {3:g} , PCP-AUPRC {4:g}, PCP-EMR {5:g}".format(eval_pre_pcp_ts, eval_rec_pcp_ts, eval_F1_pcp_ts,eval_pcp_auc,eval_pcp_auprc,eval_emr_pcp_ts))
    print("\n")
    # Predict by PCP-topK
    print("Predict by PCP-topK:")
    for top_num in range(topK):
        print("Top{0}: PCP-Precision {1:g}, PCP-Recall {2:g}, PCP-F1 {3:g}, PCP-EMR {4:g}".format(top_num+1, eval_pre_pcp_tk[top_num], eval_rec_pcp_tk[top_num], eval_F1_pcp_tk[top_num],eval_emr_pcp_tk[top_num]))
    print("\n")
    # Predict by PCP-tresholding per layer
    print("PCP-Thresholding Prediction by Layer:")
    for i in range(len(eval_pcp_metrics_per_layer)):
        print("Layer{0}: Precision {1:g}, Recall {2:g}, F1 {3:g}, AUC {4:g}, AUPRC {5:g}, EMR {6:g}".format(i+1, eval_pcp_metrics_per_layer[i]['pre'], eval_pcp_metrics_per_layer[i]['rec'], eval_pcp_metrics_per_layer[i]['f1'],eval_pcp_metrics_per_layer[i]['auc'],eval_pcp_metrics_per_layer[i]['auprc'],eval_pcp_metrics_per_layer[i]['emr']))

    return metric_dict

def eval_exact_match_ratio(true_labels_batch, predicted_labels_batch):
    """
    Computes the exact match ratio (accuracy) over a batch of true and predicted labels.

    Parameters:
    true_labels_batch (torch.Tensor): Tensor of true labels for each sample in the batch.
    predicted_labels_batch (torch.Tensor): Tensor of predicted labels for each sample in the batch.

    Returns:
    float: Exact match ratio (accuracy) averaged over the batch.
    """
    if true_labels_batch.size(0) != predicted_labels_batch.size(0):
        raise ValueError("Number of samples in true_labels_batch and predicted_labels_batch must be the same.")

    num_samples = true_labels_batch.size(0)
    total_correct = torch.sum(true_labels_batch == predicted_labels_batch).item()

    total_samples = num_samples * true_labels_batch.size(1)  # Assuming all samples have the same length

    exact_match_ratio = total_correct / total_samples
    return exact_match_ratio
def get_per_layer_metrics(scores,labels, num_classes_list):
    # Calculate Precision & Recall & F1 per Hierarchy-Layer
    begin = 0
    end = 0
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")
    eval_metrics_per_layer = []
    for i in range(len(num_classes_list)):
        if i == 0:
            begin = 0
            end = num_classes_list[0]
        else:
            begin += num_classes_list[i-1]
            end += num_classes_list[i]
        per_layer_pred = scores[:,begin:end]
        per_layer_labels = labels[:,begin:end]
        eval_pre_layer, eval_rec_layer,eval_f1_layer = precision_recall_f1_score(labels=per_layer_labels,binary_predictions=per_layer_pred)
        eval_emr_layer = eval_exact_match_ratio(true_labels_batch=per_layer_labels,predicted_labels_batch=per_layer_pred)
        eval_auc_layer = auroc(per_layer_pred.to(dtype=torch.float32),per_layer_labels.to(dtype=torch.long))
        eval_auprc_layer = auprc(per_layer_pred.to(dtype=torch.float32),per_layer_labels.to(dtype=torch.long))
        eval_metrics_per_layer.append({
            "pre": eval_pre_layer,
            "rec": eval_rec_layer,
            "f1": eval_f1_layer,
            "auc": eval_auc_layer,
            "auprc": eval_auprc_layer,
            "emr":eval_emr_layer
        })
    return eval_metrics_per_layer


    
def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted one-hot labels based on the topK.

    Args:
        scores: The all classes predicted scores provided by network.
        top_num: The max topK number (default: 5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network.
        threshold: The threshold (default: 0.5).
    Returns:
        predicted_labels: The predicted labels.
        predicted_scores: The predicted scores.
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network.
        top_num: The max topK number (default: 5).
    Returns:
        The predicted labels.
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores

def generate_one_hot(indices, num_classes):
    """
    Generate a one-hot encoding tensor for the given indices and num_classes.

    Args:
    - indices: Index vector
    - num_classes: Total number of classes

    Returns:
    - one_hot_tensor: One-hot encoding tensor
    """
    one_hot_tensor = torch.zeros(num_classes)
    one_hot_tensor[indices] = 1
    return one_hot_tensor

def prune_based_coherent_prediction(score, explicit_hierarchy, num_classes_list,pcp_threshold=-1.0):
    """ Evaluates the PCP Strategy. It gets a score list and explicit hierarchy and count of hierarchy nodes per layer
        as input and returns the path pruned class index list. """
    def _weighted_path_selecting(paths, score, pcp_threshold, hierarchy_depth):
        def _weigh(hierarchy_level):
            return 1 - (hierarchy_level/(hierarchy_depth+1))
    
        selected_paths = []
        for path in paths:
            path_score = sum([_weigh(hierachy_level) * math.log(score[path[hierachy_level]]) for hierachy_level in range(len(path))])
            if path_score > pcp_threshold:
                selected_paths.append(path)
        return selected_paths
        
    def _dynamic_threshold_pruning(score, explicit_hierarchy,num_classes_list):
        def _get_children(nodes,explicit_hierarchy):
            children = []
            for index in nodes:
                indices = list(np.where(explicit_hierarchy[index,:] == 1)[0])
                children.extend(indices)
            return children

        def _dynamic_filter(scores,mask=None):
            paths = []
            max_value = max(scores)
            threshold = max_value*0.8
            for i in range(len(scores)):
                if mask is None:
                    if scores[i] > threshold:
                        paths.append(i)
                else:
                    if i in mask:
                        if scores[i] > threshold:
                            paths.append(i)
            return paths

        def _get_paths(candidate_nodes):
            def _find_paths(node, path):
                paths = []
                hierarchy_range_lower = sum(num_classes_list[:len(path)])
                hierarchy_range_upper = hierarchy_range_lower+num_classes_list[len(path)-1]
                hierarchy_prev = sum(num_classes_list[:len(path)-1])
                if not (node >= hierarchy_prev and node < hierarchy_range_lower):
                    return None
                children = [i for i, value in enumerate(explicit_hierarchy[node]) if node != i and value == 1 and i in candidate_nodes and (i >= hierarchy_range_lower and i < hierarchy_range_upper)]
                if not children:  # If no children, return the current path
                    return [path]
                for child in children:
                    temp_path = _find_paths(child, path + [child])
                    if temp_path is None:
                        continue
                    paths.extend(temp_path)
                return paths

            all_paths = []
            for node in candidate_nodes:
                paths_from_node = _find_paths(node, [node])
                if paths_from_node is None:
                    continue
                all_paths.extend(paths_from_node)
            return all_paths
    
        all_nodes = []
        for h in range(len(num_classes_list)):
            if h == 0:
                selected = _dynamic_filter(score[:num_classes_list[h]])
                all_nodes.extend(selected)
            else:
                children = _get_children(selected,explicit_hierarchy)
                temp_selected = _dynamic_filter(score,children)
                all_nodes.extend(temp_selected)
        all_nodes = list(set(all_nodes))
        all_paths = _get_paths(all_nodes)
    
        return all_paths
    
    paths = _dynamic_threshold_pruning(score=score,explicit_hierarchy=explicit_hierarchy,num_classes_list=num_classes_list)
    selected_paths = _weighted_path_selecting(paths=paths,score=score,hierarchy_depth=len(num_classes_list),pcp_threshold=pcp_threshold)
    class_idx_list = list(set([node for path in selected_paths for node in path]))
    return class_idx_list

def create_metadata_file(word2vec_file, output_file):
    """
    Create the metadata file based on the corpus file (Used for the Embedding Visualization later).

    Args:
        word2vec_file: The word2vec file.
        output_file: The metadata file path.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist.")

    wv = KeyedVectors.load(word2vec_file, mmap='r')
    word2idx = dict([(k, v.index) for k, v in wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("[Warning] Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')

def load_images(image_paths):
    image_collection = io.ImageCollection(image_paths)
    return list(image_collection)

def convert_images(image_paths, target_size):
    images = load_images(image_paths)
    converted_images = []
    for image in images:
        # Load individual image

        # Convert images to RGB if they are grayscale
        if len(image.shape) == 2:
            image = color.gray2rgb(image)

        # Resize image to target size
        converted_images.append(image)
    return converted_images

def convert_resize_images(image_paths, target_size):
    images = load_images(image_paths)
    resized_images = []
    for image in images:
        # Load individual image

        # Convert images to RGB if they are grayscale
        if len(image.shape) == 2:
            image = color.gray2rgb(image)

        # Resize image to target size
        # Resize image to target size
        if image.shape != target_size:
            image = transform.resize(image, target_size)
            image = (image * 255).astype(np.uint8)
        resized_images.append(image)
    return resized_images
def convert_resize_normalize_images(image_paths, target_size):
    images = load_images(image_paths=image_paths)
    normalized_images = []
    for image in images:
        # Load individual image

        # Convert images to RGB if they are grayscale
        if len(image.shape) == 2:
            image = color.gray2rgb(image)


        # Resize image to target size
        if image.shape != target_size:
            image = transform.resize(image, target_size)
            #image = (image * 255).astype(np.uint8)
        else:
            image = image /255.0
        # Normalize the image using mean and standard deviation
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_normalized = (image - mean) / std
        normalized_images.append(image_normalized)
    return normalized_images
        


def augment_images(images,stream):
    fmt = 'I'*len(images)
    data_container = DataContainer(images,fmt)
    augmented_data_container = stream(data_container,return_torch=False)
    images = np.array(list(augmented_data_container.data))
    # Normalize the image pixel values to [0, 1]
    images_normalized = images.astype(np.float32) / 255.0

    # Normalize the image using mean and standard deviation
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    images_normalized = (images_normalized - mean) / std
    return images_normalized

def load_preprocess_augment_images(image_paths,input_size, stream):
    images = augment_images(tuple(convert_resize_images(image_paths,input_size)),stream)
    return images

def load_preprocess_images(image_paths,input_size):
    images = convert_resize_normalize_images(image_paths,input_size)
    return images





def load_word2vec_matrix(word2vec_file):
    """
    Get the word2idx dict and embedding matrix.

    Args:
        word2vec_file: The word2vec file.
    Returns:
        word2idx: The word2idx dict.
        embedding_matrix: The word2vec model matrix.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    wv = KeyedVectors.load(word2vec_file, mmap='r')

    word2idx = OrderedDict({"_UNK": 0})
    embedding_size = wv.vector_size
    for k, v in wv.vocab.items():
        word2idx[k] = v.index + 1
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in word2idx.items():
        if key == "_UNK":
            embedding_matrix[value] = [0. for _ in range(embedding_size)]
        else:
            embedding_matrix[value] = wv[key]
    return word2idx, embedding_matrix

def get_num_classes_from_hierarchy(hierarchy_dicts):
        return [len(hierarchy_dict.keys()) for hierarchy_dict in hierarchy_dicts]

def load_image_data_and_labels(input_file, hierarchy_dicts):
    """
    Load the wnk data from files, with images, generate one-hot labels via hierarchy.

    Args:
        input_file (_type_): the image-dict file with images from wikimedia and labels from wnk hierarchy.
        hierarchy (_type_): the wnk hierarchy
    """
    if not input_file.endswith('.json'):
        raise IOError("[Error] The research record is not a json file. "
                      "Please preprocess the research record into the json file.")
        
    
    def _find_labels_in_hierarchy_dicts(labels,hierarchy_dicts):
        for label in labels:
            label_dict = {}
            labels_index = []
            level = 0
            for dict in hierarchy_dicts:
                labels_index = []
                label_dict['layer-{0}'.format(level)] = []
                level +=1
            level = 0
            for dict in hierarchy_dicts:
                labels_index = []
                for key in dict.keys():
                    if key.endswith(label):
                        labels_index.append(dict[key])
                        temp_labels = key.split('_')
                        for i in range(len(temp_labels)-2,0,-1):
                            temp_key = '_'.join(temp_labels[:i+1])
                            temp_dict = hierarchy_dicts[i-1]
                            temp_label = temp_dict[temp_key]
                            label_dict['layer-{0}'.format(i-1)].append(temp_label)
                            
                        
                label_dict['layer-{0}'.format(level)].extend(labels_index)
                level+=1
        
        return label_dict    
    
    def _calc_total_class_labels(label_dict,hierarchy_dicts):
        level = 0
        total_class_labels = []
        for key,value in label_dict.items():
            for label in label_dict[key]:
                if level == 0:
                    total_class_labels.append(label)
                else:
                    total_class_label = label
                    for i in range(level):
                        total_class_label+=len(hierarchy_dicts[i].keys())
                    total_class_labels.append(total_class_label)
            level+=1
        return total_class_labels
    
    def _calc_total_classes(hierarchy_dicts):
        total_class_num = 0
        for dict in hierarchy_dicts:
            total_class_num+=len(dict.keys())
        return total_class_num
    
    def _create_onehot_labels(labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label
    
    Data = dict()
    with open(input_file) as fin:
        
        image_dict = json.load(fin)
        
        Data['file_names'] = []
        Data['onehot_labels'] = []
        Data['labels'] = []

        for file_name in image_dict.keys():
            labels = image_dict[file_name]        
            label_dict = _find_labels_in_hierarchy_dicts(labels,hierarchy_dicts)
            total_class_labels = _calc_total_class_labels(label_dict,hierarchy_dicts)
            total_class_num = _calc_total_classes(hierarchy_dicts)
            if len(total_class_labels) == 0:
                continue
            
            
            Data['file_names'].append(file_name)
            Data['onehot_labels'].append(_create_onehot_labels(total_class_labels, total_class_num))
            Data['labels'].append(total_class_labels)
            level = 0
            for key,labels in label_dict.items():
                if key not in Data.keys():
                    Data[key] = []
                Data[key].append(_create_onehot_labels(labels,len(hierarchy_dicts[level])))
                    
                level+=1

    return Data
        

def load_data_and_labels(args, input_file, word2idx: dict):
    """
    Load research data from files, padding sentences and generate one-hot labels.

    Args:
        args: The arguments.
        input_file: The research record.
        word2idx: The word2idx dict.
    Returns:
        The dict <Data> (includes the record tokenindex and record labels)
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not input_file.endswith('.json'):
        raise IOError("[Error] The research record is not a json file. "
                      "Please preprocess the research record into the json file.")

    def _token_to_index(x: list):
        result = []
        for item in x:
            if item not in word2idx.keys():
                result.append(word2idx['_UNK'])
            else:
                word_idx = word2idx[item]
                result.append(word_idx)
        return result

    def _create_onehot_labels(labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label

    Data = dict()
    with open(input_file) as fin:
        Data['id'] = []
        Data['content_index'] = []
        Data['content'] = []
        Data['section'] = []
        Data['subsection'] = []
        Data['group'] = []
        Data['subgroup'] = []
        Data['onehot_labels'] = []
        Data['labels'] = []

        for eachline in fin:
            record = json.loads(eachline)
            id = record['id']
            content = record['abstract']
            section = record['section']
            subsection = record['subsection']
            group = record['group']
            subgroup = record['subgroup']
            labels = record['labels']

            Data['id'].append(id)
            Data['content_index'].append(_token_to_index(content))
            Data['content'].append(content)
            Data['section'].append(_create_onehot_labels(section, args.num_classes_list[0]))
            Data['subsection'].append(_create_onehot_labels(subsection, args.num_classes_list[1]))
            Data['group'].append(_create_onehot_labels(group, args.num_classes_list[2]))
            Data['subgroup'].append(_create_onehot_labels(subgroup, args.num_classes_list[3]))
            Data['onehot_labels'].append(_create_onehot_labels(labels, args.total_classes))
            Data['labels'].append(labels)
        from tflearn.data_utils import pad_sequences
        Data['pad_seqs'] = pad_sequences(Data['content_index'], maxlen=args.pad_seq_len, value=0.)
    return Data




def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data.
        batch_size: The size of the data batch.
        num_epochs: The number of epochs.
        shuffle: Shuffle or not (default: True).
    Returns:
        A batch iterator for data set.
    """

    data = np.array(data,dtype=object)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def precision_recall_f1_score(binary_predictions, labels, average='micro'):
    """
    Calculate precision, recall, and F1 score for multi-class classification.
    
    Args:
    - binary_predictions (torch.Tensor): Predicted probabilities (shape: (n, num_classes)).
    - labels (torch.Tensor): Ground truth labels (shape: (n, num_classes)).
    - average (str): Type of averaging to perform ('micro' or 'macro').
    
    Returns:
    - precision (float): Precision score.
    - recall (float): Recall score.
    - f1_score (float): F1 score.
    """
    
    # True Positives, False Positives, False Negatives
    TP = torch.sum((binary_predictions == 1) & (labels == 1), dim=0).float()
    FP = torch.sum((binary_predictions == 1) & (labels == 0), dim=0).float()
    FN = torch.sum((binary_predictions == 0) & (labels == 1), dim=0).float()
    
    # Calculate precision, recall, and F1 score for each class
    precision = TP / (TP + FP + 1e-10)  # Adding epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)  # Adding epsilon to avoid division by zero
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)  # Adding epsilon to avoid division by zero
    
    # Perform averaging
    if average == 'micro':
        precision = torch.sum(TP) / (torch.sum(TP) + torch.sum(FP) + 1e-10)
        recall = torch.sum(TP) / (torch.sum(TP) + torch.sum(FN) + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    elif average == 'macro':
        precision = torch.mean(precision)
        recall = torch.mean(recall)
        f1_score = torch.mean(f1_score)
    else:
        raise ValueError("Invalid value for 'average'. Allowed values are 'micro' or 'macro'.")
    
    return precision.item(), recall.item(), f1_score.item()

def calculate_auc(y_true, y_score):
    batch_size = y_true.size(0)
    total_auc = 0.0

    for i in range(batch_size):
        n = y_true[i].size(0)
        n_pos = y_true[i].sum()
        n_neg = n - n_pos

        # Sort scores and corresponding true labels
        y_true_sorted, indices = torch.sort(y_true[i], descending=True)
        y_score_sorted = y_score[i][indices]

        # Calculate the ranks
        rank = torch.arange(1, n + 1, dtype=torch.float, device=y_true.device)
        rank_pos = rank[y_true_sorted == 1].sum()
        
        # Compute AUC
        auc = (rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        total_auc += auc.item()

    # Calculate average AUC
    average_auc = total_auc / batch_size

    return average_auc