# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/10 14:09:13
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : compute.py
# @Description : Some calculation-related algorithmic tools are implemented
import torch
import torch.nn as nn
import numpy as np
#prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # print(pred)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# # 计算混淆矩阵
# def compute_confusion_matrix(precited,expected):
#     predicted = np.array(precited,dtype = int)
#     expected = np.array(expected,dtype = int)
#     part = precited ^ expected          # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
#     pcount = np.bincount(part)         # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
#     tp_list = list(precited & expected)    # 将TP的计算结果转换为list
#     fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
#     tp = tp_list.count(1)                  # 统计TP的个数
#     fp = fp_list.count(1)                  # 统计FP的个数
#     tn = pcount[0] - tp                    # 统计TN的个数
#     fn = pcount[1] - fp                    # 统计FN的个数
#     return tp, fp, tn, fn

# # 计算常用指标
# def compute_indexes(tp, fp, tn, fn):
#     accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
#     precision = tp / (tp+fp)               # 精确率
#     recall = tp / (tp+fn)                  # 召回率
#     F1 = (2*precision*recall) / (precision+recall)    # F1
#     return accuracy, precision, recall, F1

'''
    predict  expect
TP:    1       1
FP:    1       0
TN:    0       0
FN:    0       1

'''
# 计算混淆矩阵
def compute_confusion_matrix(precited,expected):
    predicted = np.array(precited,dtype = int)
    expected = np.array(expected,dtype = int)

    tp_list = list(precited & expected)    # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
    tn_list = list(~precited & ~expected)    # 将TN的计算结果转换为list
    fn_list = list(~precited & expected)   # 将FN的计算结果转换为list

    tp = tp_list.count(1)                  # 统计TP的个数
    fp = fp_list.count(1)                  # 统计FP的个数
    tn = tn_list.count(1)                  # 统计TN的个数
    fn = fn_list.count(1)                  # 统计FN的个数
    return tp, fp, tn, fn

# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
    precision = tp / (tp+fp)               # 精确率
    recall = tp / (tp+fn)                  # 召回率
    F1 = (2*tp) / (2*tp+fp+fn)             # F1
    return accuracy, precision, recall, F1

def count_model_predict_digits(predict_digits,y_labels,poison_indices,y_target,save_path):
    """
    input: 
        predict_digits(np.ndarray):n*k
        y_labels(np.ndarray):n*1
        poison_indices(m*1)
        y_target(int)
        save_path(str)
    output:   
        res()
    which is saved and return is like:
    # res = {
        #         "y_target":1
        #         0:{y_post_prob_matrix, entropy_matrix,entropy_vector},
        #         1:{y_post_prob_matrix, entropy_matrix,entropy_vector,poison_y_post_prob_matrix, poison_entropy_matrix,
        #            poison_entropy_vector},
        #         2:{y_post_prob_matrix, entropy_matrix,entropy_vector},
        #         ...
        #         k:{y_post_prob_matrix, entropy_matrix,entropy_vector},
        #     }
    """

    # if type(predict_digits) is np.ndarray:
    #     predict_digits = torch.from_numpy(predict_digits)
    #     y_labels = torch.from_numpy(y_labels)
    

    res = {"y_target":str(y_target)}
    indexs = np.where(y_labels == y_target)[0]
    remain_indexs = list(set(indexs ) - set(poison_indices))
    poison_predicts = predict_digits[poison_indices]
    # softmax
    # n * k ---> n * k
    exp_arr = np.exp(poison_predicts)
    # n * k ---> n*1
    exp_sum = np.sum(exp_arr,axis=1).reshape(exp_arr.shape[0],1) 
    # n * k  /  n*1 ---> n*k
    poison_y_post_prob_matrix = exp_arr / exp_sum 
    # n * k * n*k ---> n*k
    poison_entropy_matrix = -1 * poison_y_post_prob_matrix * np.log(poison_y_post_prob_matrix)
    # n * k ---> n*1
    poison_entropy_vector = np.sum(poison_entropy_matrix,axis=1)

    predicts = predict_digits[remain_indexs]
    # n * k --->n * k

    # y_post_prob_matrix =  nn.functional.softmax(predicts)
    # n * k ---> n * k
    exp_arr = np.exp(predicts)
    # n * k ---> n*1
    exp_sum = np.sum(exp_arr,axis=1).reshape(exp_arr.shape[0],1)
    # n * k  /  n*1 ---> n*k
    y_post_prob_matrix = exp_arr / exp_sum 

    # n * k * n*k
    # entropy_matrix = y_post_prob_matrix * torch.log(y_post_prob_matrix)
    entropy_matrix = -1 * y_post_prob_matrix * np.log(y_post_prob_matrix)
    # n * k ---> n*1
    entropy_vector = np.sum(entropy_matrix,axis=1)
    # entropy_vector = torch.sum(entropy_matrix,1)
    label = y_target
   
    res[str(label)] = {"predicts":predicts,"y_post_prob_matrix":y_post_prob_matrix,"entropy_matrix":entropy_matrix,
                  "entropy_vector":entropy_vector,"poison_predicts":poison_predicts,"poison_y_post_prob_matrix":poison_y_post_prob_matrix,
                  "poison_entropy_matrix":  poison_entropy_matrix, "poison_entropy_vector": poison_entropy_vector
                }
    for label in range(10):
        if label ==  y_target:
            continue
        indexs = np.where(y_labels == label)[0]
        predicts = predict_digits[indexs]
        # n * k ---> n * k
        exp_arr = np.exp(predicts)
        # n * k ---> n*1
        exp_sum = np.sum(exp_arr,axis=1).reshape(exp_arr.shape[0],1)
        # n * k --->n * k
        y_post_prob_matrix =  exp_arr / exp_sum
        # n * k * n*k
        entropy_matrix = -1 * y_post_prob_matrix * np.log(y_post_prob_matrix)
        # n * k ---> n
        entropy_vector = np.sum(entropy_matrix,axis=1)
        res[str(label)] = {"predicts":predicts,"y_post_prob_matrix":y_post_prob_matrix,"entropy_matrix":entropy_matrix,"entropy_vector":entropy_vector}
    np.savez(save_path,**res)
    return res
