# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:50:10 2019

@author: yinr0002
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import warnings
sys.path.append(os.path.abspath("C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/5_Journal/Bioinformatics_2/code/"))
#sys.path.append(os.path.abspath("/content/drive/My Drive/Colab Notebooks/bioinformatics/code"))

from data_generation import cnn_training_data
from model import train_test_split_data
from model import knn_cross_validation
from model import svm_cross_validation
from model import logistic_cross_validation
from model import randomforest_cross_validation
from model import bayes_cross_validation
from model import CNN_H1N1
from model import CNN_H3N2
from model import CNN_H5N1
from model import SENet18
from model import IVA_CNN
from model import svm_baseline
from model import rf_baseline
from model import lr_baseline
from model import knn_baseline
from model import nn_baseline
from model import reshape_to_linear
from model import setup_seed
from train_cnn import train_cnn
from method_comparison import get_confusion_matrix
from method_comparison import distance_mutation
from method_comparison import get_accuracy
from method_comparison import get_precision
from method_comparison import get_recall
from method_comparison import get_f1score
from method_comparison import get_mcc


warnings.filterwarnings('ignore')
os.chdir("C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/5_Journal/Bioinformatics_2/data")
#os.chdir("/content/drive/My Drive/Colab Notebooks/bioinformatics/data")

np.random.seed(10)

#prepare for H1N1 data in epitope features
H1N1_epitope_data = pd.read_csv('training/H1N1_epitope_data.csv')
H1N1_epitope_data = H1N1_epitope_data.iloc[:,1:7]
H1N1_epitope_feature = H1N1_epitope_data.iloc[:, H1N1_epitope_data.columns!='label']
H1N1_epitope_label = H1N1_epitope_data['label']

##prepare for H1N1 data in regional features
#H1N1_regional_data = pd.read_csv('training/H1N1_regional_data.csv')
#H1N1_regional_data = H1N1_regional_data.iloc[:,1:12]
#H1N1_regional_feature = H1N1_regional_data.iloc[:, H1N1_regional_data.columns!='label']
#H1N1_regional_label = H1N1_regional_data['label']


##prepare for H3N2 data in epitope features
#H3N2_epitope_data = pd.read_csv('training/H3N2_epitope_data.csv')
#H3N2_epitope_data = H3N2_epitope_data.iloc[:,1:7]
#H3N2_epitope_feature = H3N2_epitope_data.iloc[:, H3N2_epitope_data.columns!='label']
#H3N2_epitope_label = H3N2_epitope_data['label']
##prepare for H3N2 data in regional features
#H3N2_regional_data = pd.read_csv('training/H3N2_regional_data.csv')
#H3N2_regional_data = H3N2_regional_data.iloc[:,1:12]
#H3N2_regional_feature = H3N2_regional_data.iloc[:, H3N2_regional_data.columns!='label']
#H3N2_regional_label = H3N2_regional_data['label']
#
##prepare for H5N1 data in epitope features
#H5N1_epitope_data = pd.read_csv('training/H5N1_epitope_data.csv')
#H5N1_epitope_data = H5N1_epitope_data.iloc[:,1:7]
#H5N1_epitope_feature = H5N1_epitope_data.iloc[:, H5N1_epitope_data.columns!='label']
#H5N1_epitope_label = H5N1_epitope_data['label']
##prepare for H5N1 data in regional features
#H5N1_regional_data = pd.read_csv('training/H5N1_regional_data.csv')
#H5N1_regional_data = H5N1_regional_data.iloc[:,1:12]
#H5N1_regional_feature = H5N1_regional_data.iloc[:, H5N1_regional_data.columns!='label']
#H5N1_regional_label = H5N1_regional_data['label']


def main():
    parameters = {
            
      # select influenza subtype
      'subtype': subtype,
      
      # select the way for feature generation
      'feature_type': feature_type,
    
      # 'rf', lr', 'knn', 'svm', 'cnn'
      #'model': model,
    
      # Number of hidden units in the encoder
      'hidden_size': 128,
    
      # Droprate (applied at input)
      'dropout_p': 0.5,
    
      # Note, no learning rate decay implemented
      'learning_rate': 0.001,
    
      # Size of mini batch
      'batch_size': 32,
    
      # Number of training iterations
      'num_of_epochs': 100
    }
    
    if parameters['subtype'] == 'H1N1':
        #read antigenic data and sequence data
        H1N1_Antigenic_dist = pd.read_csv('antigenic/H1N1_antigenic.csv')
        H1N1_seq = pd.read_csv('sequence/H1N1/H1N1_sequence_HA1.csv', names=['seq', 'description'])
        
        if model_mode == 'Tradition model': 
            if feature_type == 'Min-Shi Lee':
                print('\n')
                H1N1_num_mut_list = distance_mutation(H1N1_Antigenic_dist, H1N1_seq)    
                H1N1_Antigenic_dist_list = list(H1N1_Antigenic_dist['Distance'])
                conf_matrix = get_confusion_matrix(H1N1_num_mut_list, H1N1_Antigenic_dist_list, 'H1N1')        
                H1N1_acc = get_accuracy(conf_matrix)
                H1N1_pre = get_precision(conf_matrix)
                H1N1_rec = get_recall(conf_matrix)
                H1N1_f1 = get_f1score(conf_matrix)
                H1N1_mcc = get_mcc(conf_matrix)
                print('Min-Shi Lee method on H1N1:')
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(H1N1_acc, H1N1_pre, H1N1_rec, H1N1_f1, H1N1_mcc))
                
            elif feature_type == 'Yu-Chieh Liao':
                print('\n')
                H1N1_data = pd.read_csv('training/Yu-Chieh Liao/H1N1.csv')
                H1N1_data = H1N1_data.iloc[:,1:329]
                H1N1_feature = H1N1_data.iloc[:, H1N1_data.columns!='label']
                H1N1_label = H1N1_data['label']
                print('Yu-Chieh Liao method on H1N1 using svm:')
                #train_x, test_x, train_y, test_y = train_test_split_data(H1N1_feature, H1N1_label, 0.2)
                svm_cross_validation(H1N1_feature, H1N1_label)

            elif feature_type == 'William Lees':
                print('\n')
                H1N1_data = pd.read_csv('training/William Lees/H1N1_new_epitope_data.csv')
                H1N1_data = H1N1_data.iloc[:,1:7]
                H1N1_feature = H1N1_data.iloc[:, H1N1_data.columns!='label']
                H1N1_label = H1N1_data['label']
                print('William Lees method on H1N1 using svm:')
                logistic_cross_validation(H1N1_feature, H1N1_label)
                
            elif feature_type == 'Peng Yousong':
                print('\n')
                H1N1_data = pd.read_csv('training/Peng Yousong/H1N1_regional_data.csv')
                H1N1_data = H1N1_data.iloc[:,1:12]
                H1N1_feature = H1N1_data.iloc[:, H1N1_data.columns!='label']
                H1N1_label = H1N1_data['label']
                print('Peng Yousong method on H1N1 using naive bayes:')
                bayes_cross_validation(H1N1_feature, H1N1_label)
                
            elif feature_type == 'Yuhua Yao':
                print('\n')
                H1N1_data = pd.read_csv('training/Yuhua Yao/H1N1.csv')
                H1N1_data = H1N1_data.iloc[:,1:329]
                H1N1_feature = H1N1_data.iloc[:, H1N1_data.columns!='label']
                H1N1_label = H1N1_data['label']
                print('Yuhua Yao method on H1N1 using random forest:')
                randomforest_cross_validation(H1N1_feature, H1N1_label)
                    
        elif model_mode == 'Deep model':
            setup_seed(10)
            
            #feature generation
            feature, label = cnn_training_data(H1N1_Antigenic_dist, H1N1_seq)
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2)
            
            if baseline == 'rf_baseline':
                print('rf_baseline + ProVect on H1N1:')
                rf_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'lr_baseline':
                print('lr_baseline + ProVect on H1N1:')
                lr_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'svm_baseline':
                print('svm_baseline + ProVect on H1N1:')
                svm_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'knn_baseline':
                print('knn_baseline + ProVect on H1N1:')
                knn_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'nn_baseline':
                print('nn_baseline + ProVect on H1N1:')
                nn_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'cnn':
                print('CNN + ProVect on H1N1:')
                train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x =  np.reshape(test_x, (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                print(np.array(train_x).shape)
                print(np.array(test_x).shape)
            
                train_x = torch.tensor(train_x, dtype=torch.float32)
                train_y = torch.tensor(train_y, dtype=torch.int64)
                test_x = torch.tensor(test_x, dtype=torch.float32)
                test_y = torch.tensor(test_y, dtype=torch.int64)
                net = CNN_H1N1()
#                if torch.cuda.is_available():
#                    print('running with GPU')
#                    net.cuda()
                
                train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y)
            elif baseline == 'iva-cnn':
                print('iva-cnn + ProVect on H1N1:')
                train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x =  np.reshape(test_x, (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                print(np.array(train_x).shape)
                print(np.array(test_x).shape)
            
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

                net = IVA_CNN(1, 128, 2, 2)
                net.cuda()
                train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y)            
            
    elif parameters['subtype'] == 'H3N2':
        H3N2_Antigenic_dist = pd.read_csv('antigenic/H3N2_antigenic.csv')
        H3N2_seq = pd.read_csv('sequence/H3N2/H3N2_sequence_HA1.csv', names=['seq', 'description'])
        if model_mode == 'Tradition model': 
            if feature_type == 'Min-Shi Lee':
                print('\n')
                H3N2_num_mut_list = distance_mutation(H3N2_Antigenic_dist, H3N2_seq)    
                H3N2_Antigenic_dist_list = list(H3N2_Antigenic_dist['Distance'])
                conf_matrix = get_confusion_matrix(H3N2_num_mut_list, H3N2_Antigenic_dist_list, 'H3N2')        
                H3N2_acc = get_accuracy(conf_matrix)
                H3N2_pre = get_precision(conf_matrix)
                H3N2_rec = get_recall(conf_matrix)
                H3N2_f1 = get_f1score(conf_matrix)
                H3N2_mcc = get_mcc(conf_matrix)
                print('Min-Shi Lee method on H3N2:')
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(H3N2_acc, H3N2_pre, H3N2_rec, H3N2_f1, H3N2_mcc))
                
            elif feature_type == 'Yu-Chieh Liao':
                print('\n')
                H3N2_data = pd.read_csv('training/Yu-Chieh Liao/H3N2.csv')
                H3N2_data = H3N2_data.iloc[:,1:331]
                H3N2_feature = H3N2_data.iloc[:, H3N2_data.columns!='label']
                H3N2_label = H3N2_data['label']
                print('Yu-Chieh Liao method on H3N2 using svm:')
                #train_x, test_x, train_y, test_y = train_test_split_data(H3N2_feature, H3N2_label, 0.2)
                svm_cross_validation(H3N2_feature, H3N2_label)

            elif feature_type == 'William Lees':
                print('\n')
                H3N2_data = pd.read_csv('training/William Lees/H3N2_new_epitope_data.csv')
                H3N2_data = H3N2_data.iloc[:,1:7]
                H3N2_feature = H3N2_data.iloc[:, H3N2_data.columns!='label']
                H3N2_label = H3N2_data['label']
                print('William Lees method on H3N2 using svm:')
                logistic_cross_validation(H3N2_feature, H3N2_label)
                
            elif feature_type == 'Peng Yousong':
                print('\n')
                H3N2_data = pd.read_csv('training/Peng Yousong/H3N2_regional_data.csv')
                H3N2_data = H3N2_data.iloc[:,1:12]
                H3N2_feature = H3N2_data.iloc[:, H3N2_data.columns!='label']
                H3N2_label = H3N2_data['label']
                print('Peng Yousong method on H3N2 using naive bayes:')
                bayes_cross_validation(H3N2_feature, H3N2_label)
                
            elif feature_type == 'Yuhua Yao':
                print('\n')
                H3N2_data = pd.read_csv('training/Yuhua Yao/H3N2.csv')
                H3N2_data = H3N2_data.iloc[:,1:331]
                H3N2_feature = H3N2_data.iloc[:, H3N2_data.columns!='label']
                H3N2_label = H3N2_data['label']
                print('Yuhua Yao method on H3N2 using random forest:')
                randomforest_cross_validation(H3N2_feature, H3N2_label)
                    
        elif model_mode == 'Deep model':
            setup_seed(20)

            feature, label = cnn_training_data(H3N2_Antigenic_dist, H3N2_seq)
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2)
            
            if baseline == 'rf_baseline':
                print('rf_baseline + ProVect on H3N2:')
                rf_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'lr_baseline':
                print('lr_baseline + ProVect on H3N2:')
                lr_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'svm_baseline':
                print('svm_baseline + ProVect on H3N2:')
                svm_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'knn_baseline':
                print('knn_baseline + ProVect on H3N2:')
                knn_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'nn_baseline':
                print('nn_baseline + ProVect on H3N2:')
                nn_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'cnn':
                print('CNN + ProVect on H3N2:')
                train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x =  np.reshape(test_x, (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                print(np.array(train_x).shape)
                print(np.array(test_x).shape)
            
                train_x = torch.tensor(train_x, dtype=torch.float32)
                train_y = torch.tensor(train_y, dtype=torch.int64)
                test_x = torch.tensor(test_x, dtype=torch.float32)
                test_y = torch.tensor(test_y, dtype=torch.int64)
                net = CNN_H3N2()
#                if torch.cuda.is_available():
#                    print('running with GPU')
#                    net.cuda()
                
                train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y)
            elif baseline == 'iva-cnn':
                print('iva-cnn + ProVect on H3N2:')
                train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x =  np.reshape(test_x, (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                print(np.array(train_x).shape)
                print(np.array(test_x).shape)
            
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

                net = IVA_CNN(1, 128, 2, 2)
                net.cuda()
                train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y)     


    elif parameters['subtype'] == 'H5N1':
        H5N1_Antigenic_dist = pd.read_csv('antigenic/H5N1_antigenic.csv')
        H5N1_seq = pd.read_csv('sequence/H5N1/H5N1_sequence_HA1.csv', names=['seq', 'description'])
        if model_mode == 'Tradition model': 
            if feature_type == 'Min-Shi Lee':
                print('\n')
                H5N1_num_mut_list = distance_mutation(H5N1_Antigenic_dist, H5N1_seq)    
                H5N1_Antigenic_dist_list = list(H5N1_Antigenic_dist['Distance'])
                conf_matrix = get_confusion_matrix(H5N1_num_mut_list, H5N1_Antigenic_dist_list, 'H5N1')        
                H5N1_acc = get_accuracy(conf_matrix)
                H5N1_pre = get_precision(conf_matrix)
                H5N1_rec = get_recall(conf_matrix)
                H5N1_f1 = get_f1score(conf_matrix)
                H5N1_mcc = get_mcc(conf_matrix)
                print('Min-Shi Lee method on H5N1:')
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'%(H5N1_acc, H5N1_pre, H5N1_rec, H5N1_f1, H5N1_mcc))
                
            elif feature_type == 'Yu-Chieh Liao':
                print('\n')
                H5N1_data = pd.read_csv('training/Yu-Chieh Liao/H5N1.csv')
                H5N1_data = H5N1_data.iloc[:,1:322]
                H5N1_feature = H5N1_data.iloc[:, H5N1_data.columns!='label']
                H5N1_label = H5N1_data['label']
                print('Yu-Chieh Liao method on H5N1 using svm:')
                #train_x, test_x, train_y, test_y = train_test_split_data(H5N1_feature, H5N1_label, 0.2)
                svm_cross_validation(H5N1_feature, H5N1_label)

            elif feature_type == 'William Lees':
                print('\n')
                H5N1_data = pd.read_csv('training/William Lees/H5N1_new_epitope_data.csv')
                H5N1_data = H5N1_data.iloc[:,1:7]
                H5N1_feature = H5N1_data.iloc[:, H5N1_data.columns!='label']
                H5N1_label = H5N1_data['label']
                print('William Lees method on H5N1 using svm:')
                logistic_cross_validation(H5N1_feature, H5N1_label)
                
            elif feature_type == 'Peng Yousong':
                print('\n')
                H5N1_data = pd.read_csv('training/Peng Yousong/H3N2_regional_data.csv')
                H5N1_data = H5N1_data.iloc[:,1:12]
                H5N1_feature = H5N1_data.iloc[:, H5N1_data.columns!='label']
                H5N1_label = H5N1_data['label']
                print('Peng Yousong method on H5N1 using naive bayes:')
                bayes_cross_validation(H5N1_feature, H5N1_label)
                
            elif feature_type == 'Yuhua Yao':
                print('\n')
                H5N1_data = pd.read_csv('training/Yuhua Yao/H5N1.csv')
                H5N1_data = H5N1_data.iloc[:,1:322]
                H5N1_feature = H5N1_data.iloc[:, H5N1_data.columns!='label']
                H5N1_label = H5N1_data['label']
                print('Yuhua Yao method on H5N1 using random forest:')
                randomforest_cross_validation(H5N1_feature, H5N1_label)
                    
        elif model_mode == 'Deep model':
            setup_seed(20)

            feature, label = cnn_training_data(H5N1_Antigenic_dist, H5N1_seq)
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2)
            
            if baseline == 'rf_baseline':
                print('rf_baseline + ProVect on H5N1:')
                rf_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'lr_baseline':
                print('lr_baseline + ProVect on H5N1:')
                lr_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'svm_baseline':
                print('svm_baseline + ProVect on H5N1:')
                svm_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'knn_baseline':
                print('knn_baseline + ProVect on H5N1:')
                knn_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'nn_baseline':
                print('nn_baseline + ProVect on H5N1:')
                nn_baseline(reshape_to_linear(train_x), train_y, reshape_to_linear(test_x), test_y)
            elif baseline == 'cnn':
                print('CNN + ProVect on H5N1:')
                train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x =  np.reshape(test_x, (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                print(np.array(train_x).shape)
                print(np.array(test_x).shape)
            
                train_x = torch.tensor(train_x, dtype=torch.float32)
                train_y = torch.tensor(train_y, dtype=torch.int64)
                test_x = torch.tensor(test_x, dtype=torch.float32)
                test_y = torch.tensor(test_y, dtype=torch.int64)
                net = CNN_H5N1()
#                if torch.cuda.is_available():
#                    print('running with GPU')
#                    net.cuda()
                
                train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y)
            elif baseline == 'iva-cnn':
                print('iva-cnn + ProVect on H5N1:')
                train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x =  np.reshape(test_x, (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                print(np.array(train_x).shape)
                print(np.array(test_x).shape)
            
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

                net = IVA_CNN(1, 128, 2, 2)
                net.cuda()
                train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y)     



if __name__ == '__main__':
    influenza_type = ['H1N1', 'H3N2', 'H5N1']
    method = ['Deep model', 'Tradition model']
    #feature_engineering = ['Min-Shi Lee', 'Yu-Chieh Liao', 'William Lees', 'Peng Yousong', 'Yuhua Yao']
    H5N1_Antigenic_dist = pd.read_csv('antigenic/H5N1_antigenic.csv')
    H5N1_seq = pd.read_csv('sequence/H5N1/H5N1_sequence_HA1.csv', names=['seq', 'description'])
    feature, label = cnn_training_data(H5N1_Antigenic_dist, H5N1_seq)
    subtype = influenza_type[2]
    model_mode = method[0]
    feature_type = ['Min-Shi Lee', 'Yu-Chieh Liao', 'William Lees', 'Peng Yousong', 'Yuhua Yao']
    #baseline = ['svm_baseline', 'rf_baseline', 'lr_baseline', 'knn_baseline', 'nn_baseline', 'cnn']
    baseline = ['cnn']
    
    if model_mode == 'Deep model':
        for baseline in baseline:
            main()
    elif model_mode == 'Tradition model':
        for feature_type in feature_type:
            main()
        

        

##################################################################################################################     
##test on different classifiers of influenza datasets on distinct feature transformation          
#    if parameters['subtype'] == 'H1N1':
#        if model_mode == 'Tradition model': 
#            if parameters['feature_type'] == 'epitope':
#                print('\n')
#                if parameters['model'] == 'rf':
#                    print('Ranodm forest using epitope features on H1N1:')
#                    randomforest_cross_validation(H1N1_epitope_feature, H1N1_epitope_label)
#                elif parameters['model'] == 'lr':
#                    print('Logistic regression using epitope features on H1N1:')
#                    logistic_cross_validation(H1N1_epitope_feature, H1N1_epitope_label)
#                elif parameters['model'] == 'knn':
#                    print('KNN using epitope features on H1N1:')
#                    knn_cross_validation(H1N1_epitope_feature, H1N1_epitope_label)
#                elif parameters['model'] == 'svm':
#                    print('SVM using epitope features on H1N1:')
#                    svm_cross_validation(H1N1_epitope_feature, H1N1_epitope_label)
#                
#            elif feature_type == 'regional':
#                print('\n')
#                if parameters['model'] == 'rf':
#                    print('Ranodm forest using regional features on H1N1:')
#                    randomforest_cross_validation(H1N1_epitope_feature, H1N1_epitope_label)
#                elif parameters['model'] == 'lr':
#                    print('Logistic regression using regional features on H1N1:')
#                    logistic_cross_validation(H1N1_epitope_feature, H1N1_epitope_label)
#                elif parameters['model'] == 'knn':
#                    print('KNN using regional features on H1N1:')
#                    knn_cross_validation(H1N1_epitope_feature, H1N1_epitope_label)
#                elif parameters['model'] == 'svm':
#                    print('SVM using regional features on H1N1:')
#                    svm_cross_validation(H1N1_epitope_feature, H1N1_epitope_label)            
            
