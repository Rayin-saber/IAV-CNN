# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:20:17 2019

@author: YINR0002
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import pandas as pd
import numpy as np
import scipy as sp
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import fasta_to_csv
from model import calculate_label
from model import generate_feature
from model import strain_selection
from model import replace_uncertain_amino_acids
from model import train_test_split_data

#os.chdir('/content/drive/My Drive/Colab Notebooks/bioinformatics/data')
os.chdir('C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/5_Journal/Bioinformatics_2/data/')

H1N1_seq = pd.read_csv('sequence/H1N1/H1N1_sequence_HA1.csv', names=['seq', 'description'])
H1N1_Antigenic_dist = pd.read_csv('antigenic/H1N1_antigenic.csv')
H3N2_seq = pd.read_csv('sequence/H3N2/H3N2_sequence_HA1.csv', names=['seq', 'description'])
H3N2_Antigenic_dist = pd.read_csv('antigenic/H3N2_antigenic.csv')
H5N1_seq = pd.read_csv('sequence/H5N1/H5N1_sequence_HA1.csv', names=['seq', 'description'])
H5N1_Antigenic_dist = pd.read_csv('antigenic/H5N1_antigenic.csv')

####################################################################################################################################
#Min-Shi Lee's method (Predicting Antigenic Variants of Influenza A/H3N2 Viruses)
def distance_mutation(distance_input, seq_input):
    num_mut_list = []
    for i in range(0, distance_input.shape[0]):
        mutation_count = 0
        strain_1 = []
        strain_2 = []
        for j in range(0, seq_input.shape[0]):
            if seq_input['description'].iloc[j] == distance_input['Strain1'].iloc[i]:
                strain_1 = seq_input['seq'].iloc[j]
            if seq_input['description'].iloc[j] == distance_input['Strain2'].iloc[i]:
                strain_2 = seq_input['seq'].iloc[j]
        mutation_count = sum(0 if c1 == c2 else 1 for c1, c2 in zip(strain_1, strain_2))
        num_mut_list.append(mutation_count)
    return num_mut_list
        
H1N1_num_mut_list = distance_mutation(H1N1_Antigenic_dist, H1N1_seq)    
H1N1_Antigenic_dist_list = list(H1N1_Antigenic_dist['Distance'])

H3N2_num_mut_list = distance_mutation(H3N2_Antigenic_dist, H3N2_seq)    
H3N2_antigenic_dist_list = list(H3N2_Antigenic_dist['Distance'])

H5N1_num_mut_list = distance_mutation(H5N1_Antigenic_dist, H5N1_seq)    
H5N1_antigenic_dist_list = list(H5N1_Antigenic_dist['Distance'])
#plt.scatter(num_mut_list, y_pred, c='b')

def get_confusion_matrix(x_true, y_true, subtype):
    TP, FP, TN, FN = 0, 0, 0, 0
    threshold = 0
    #optimized threshold (with best performance)
    if subtype == 'H1N1':
        threshold = 11
    elif subtype == 'H3N2':
        threshold = 9
    elif subtype == 'H5N1':
        threshold = 12
        
    for i in range(len(y_true)):
        if x_true[i] <= threshold:
            if y_true[i] < threshold:
                TN = TN + 1
            else:
                FP = FP + 1
        else:
            if y_true[i] >= threshold:
                TP = TP + 1
            else:
                FN = FN + 1
                
    conf_matrix = [
        [FP, TP],
        [TN, FN]
    ]

    return conf_matrix

def get_accuracy(conf_matrix):
    """
    Calculates accuracy metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][1], conf_matrix[0][0], conf_matrix[1][1], conf_matrix[1][0]
    return (TP + TN) / (TP + FP + FN + TN)

def get_precision(conf_matrix):
    """
    Calculates precision metric from the given confusion matrix.
    """
    TP, FP = conf_matrix[0][1], conf_matrix[0][0]

    if TP + FP > 0:
        return TP / (TP + FP)
    else:
        return 0

def get_recall(conf_matrix):
    """
    Calculates recall metric from the given confusion matrix.
    """
    TP, FN = conf_matrix[0][1], conf_matrix[1][1]

    if TP + FN > 0:
        return TP / (TP + FN)
    else:
        return 0
    
def get_f1score(conf_matrix):
    """
    Calculates f1-score metric from the given confusion matrix.
    """
    p = get_precision(conf_matrix)
    r = get_recall(conf_matrix)

    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0
    
def get_mcc(conf_matrix):
    """
    Calculates Matthew's Correlation Coefficient metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][1], conf_matrix[0][0], conf_matrix[1][1], conf_matrix[1][0]
    if TP + FP > 0 and TP + FN > 0 and TN + FP > 0 and TN + FN > 0:
        return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        return 0
        
#conf_matrix = get_confusion_matrix(H1N1_num_mut_list, H1N1_Antigenic_dist_list, 'H1N1')        
#H1N1_acc = get_accuracy(conf_matrix)
#H1N1_pre = get_precision(conf_matrix)
#H1N1_rec = get_recall(conf_matrix)
#H1N1_f1 = get_f1score(conf_matrix)
#H1N1_mcc = get_mcc(conf_matrix)
#print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(H1N1_acc, H1N1_pre, H1N1_rec, H1N1_f1, H1N1_mcc))
#
#conf_matrix = get_confusion_matrix(H3N2_num_mut_list, H3N2_antigenic_dist_list, 'H3N2')        
#H3N2_acc = get_accuracy(conf_matrix)
#H3N2_pre = get_precision(conf_matrix)
#H3N2_rec = get_recall(conf_matrix)
#H3N2_f1 = get_f1score(conf_matrix)
#H3N2_mcc = get_mcc(conf_matrix)
#print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(H3N2_acc, H3N2_pre, H3N2_rec, H3N2_f1, H3N2_mcc))
#
#conf_matrix = get_confusion_matrix(H5N1_num_mut_list, H5N1_antigenic_dist_list, 'H5N1')        
#H5N1_acc = get_accuracy(conf_matrix)
#H5N1_pre = get_precision(conf_matrix)
#H5N1_rec = get_recall(conf_matrix)
#H5N1_f1 = get_f1score(conf_matrix)
#H5N1_mcc = get_mcc(conf_matrix)
#print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(H5N1_acc, H5N1_pre, H5N1_rec, H5N1_f1, H5N1_mcc))
####################################################################################################################################







































