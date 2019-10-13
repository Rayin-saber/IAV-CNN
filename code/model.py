# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:05:12 2019

@author: yinr0002
"""
import os, sys
import pandas as pd
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from sklearn import datasets
from sklearn import neighbors
from sklearn import svm
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier 
from matplotlib import pyplot 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from imblearn.metrics import geometric_mean_score
from scipy import interp
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from validation import evaluate

#os.chdir('/content/drive/My Drive/Colab Notebooks/bioinformatics/data/')
os.chdir('C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/5_Journal/Bioinformatics_2/data/')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

#convert fasta file to csv
def fasta_to_csv(input_file):
    flu = open(os.path.expanduser("sequence/H1N1_sequence.csv"), 'w')
    for seq_record in input_file:
        flu_seq = ''
        #record_id = seq_record.id
        seq = seq_record.seq
        description = seq_record.description
        flu_seq =  str(seq) + ',' + str(description) + "\n"
        flu.write(flu_seq)
    flu.close()

#convert distance values into label, distance>=4 (label=1) distance<4 (label=0)
def calculate_label(antigenic_data):
    distance_label = []
    if len(set(antigenic_data['Distance'])) == 2:
        for i in range(0, antigenic_data.shape[0]):
            if antigenic_data['Distance'].iloc[i] == 1:
                distance_label.append(1)
            elif antigenic_data['Distance'].iloc[i] == 0:
                distance_label.append(0)
            else:
                print('error')
    else:
        for i in range(0, antigenic_data.shape[0]):
            if antigenic_data['Distance'].iloc[i] >= 4.0:
                distance_label.append(1)
            else:
                distance_label.append(0)   
    return distance_label


#feature generation for epitope and regional method
def generate_feature(region_type, distance_input, seq_input):
    distance_label = calculate_label(distance_input)
    label = {'label': distance_label}
    label = pd.DataFrame(label)
    
    index = pd.Series(np.arange(distance_input.shape[0]))
    if len(region_type) == 2:
        columns = ['epitope_a', 'epitope_b']
    elif len(region_type) == 5:
        columns = ['new_epitope_a', 'new_epitope_b', 'new_epitope_c', 'new_epitope_d', 'new_epitope_e']
    elif len(region_type) == 10:
        columns = ['regional_1', 'regional_2', 'regional_3', 'regional_4', 'regional_5', 
                   'regional_6', 'regional_7', 'regional_8', 'regional_9', 'regional_10',]
    Mut_feature = pd.DataFrame(index=index, columns=columns)
    
    for region in region_type:
        site = region_type[region]
        for i in range(0, distance_input.shape[0]):
            mut_count = 0
            for a in range(0, len(site)):
                value_1 = 0
                value_2 = 0
                for j in range(0, seq_input.shape[0]):
                    if seq_input['description'].iloc[j].upper() == distance_input['Strain1'].iloc[i].upper() :
                        value_1 = seq_input['seq'].iloc[j][site[a] - 1]   #index is 0 by default, but the site starts from 1
                    if seq_input['description'].iloc[j].upper()  == distance_input['Strain2'].iloc[i].upper() :
                        value_2 = seq_input['seq'].iloc[j][site[a] - 1]
                if value_1 == value_2:
                    mut_count = mut_count + 0
                else:
                    mut_count = mut_count + 1
        
            Mut_feature[region].iloc[i] = mut_count
    Mut_feature = Mut_feature.join(label)   
    return(Mut_feature)


#extract sequences to construct new dataset
def strain_selection(distance_data, seq_data):
    raw_data = []
    strain1 = []
    strain2 = []
    label = calculate_label(distance_data)
    for i in range(0, distance_data.shape[0]):
        seq1 = []
        seq2 = []
        flag1 = 0 
        flag2 = 0
        for j in range(0, seq_data.shape[0]):
            if str(seq_data['description'].iloc[j]).upper() == str(distance_data['Strain1'].iloc[i]).upper():
                seq1 = str(seq_data['seq'].iloc[j]).upper()
                flag1 = 1
            if str(seq_data['description'].iloc[j]).upper() == str(distance_data['Strain2'].iloc[i]).upper():
                seq2 = str(seq_data['seq'].iloc[j]).upper()
                flag2 = 1
            if flag1 == 1 and flag2 == 1:
                break
        strain1.append(seq1)
        strain2.append(seq2)
    
    raw_data.append(strain1)
    raw_data.append(strain2)
    raw_data.append(label)
    return raw_data


def replace_uncertain_amino_acids(amino_acids):
    
    """
    Randomly selects replacements for all uncertain amino acids.
    Expects and returns a string.
    """
    replacements = {'B': 'DN',
                  'J': 'IL',
                  'Z': 'EQ',
                  'X': 'ACDEFGHIKLMNPQRSTVWY'}

    for uncertain in replacements.keys():
        
        amino_acids = amino_acids.replace(uncertain, random.choice(replacements[uncertain]))

    return amino_acids

#split data into training and testing
def train_test_split_data(feature, label, split_ratio):
    setup_seed(20)
    train_x, test_x, train_y, test_y = [], [], [], []
    feature_new, label_new = [], []
    num_of_training = int(math.floor(len(feature) * (1 - split_ratio)))
    
    shuffled_index = np.arange(len(feature))
    random.shuffle(shuffled_index)
    for i in range(0, len(feature)):
        feature_new.append(feature[shuffled_index[i]]) 
        label_new.append(label[shuffled_index[i]])
  
    train_x = feature_new[:num_of_training]
    train_y = label_new[:num_of_training]
    test_x = feature_new[num_of_training:]
    test_y = label_new[num_of_training:]
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    return train_x, test_x, train_y, test_y

def reshape_to_linear(x):
    
  output = np.reshape(x, (x.shape[0], -1))
  
  return output


def knn_cross_validation(train_x, train_y):
    np.random.seed(100)
    clf = neighbors.KNeighborsClassifier()
    
    #calculate the accuracy
    accuracy = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy') 
    print("accuracy: %f" %accuracy.mean() + '\n')
    #print accuracy
    
    #calculate the precision
    precision = cross_val_score(clf, train_x, train_y, cv=5, scoring='precision_macro')
    print("precision: %f" %precision.mean() + '\n')
    
    #calculate the recall score
    recall = cross_val_score(clf, train_x, train_y, cv=5, scoring='recall_macro')
    print("recall: %f" %recall.mean() + '\n')
    
    #calculate the f_measure
    f_measure = cross_val_score(clf, train_x, train_y, cv=5, scoring='f1_macro')
    print("f_measure: %f " %f_measure.mean() + '\n')
    
    
    #generate classification report and MCC and G-mean value
    y_pred = cross_val_predict(clf, train_x, train_y, cv=5)
    G_mean = geometric_mean_score(train_y, y_pred)
    MCC = matthews_corrcoef(train_y, y_pred)
    print("G_mean: %f" %G_mean.mean() + '\n')
    print("MCC: %f" %np.mean(MCC) + '\n')
    
    print("Classification_report:")
    print(metrics.classification_report(train_y, y_pred))
    

def svm_cross_validation(train_x, train_y):
    np.random.seed(100)
    clf = svm.SVC()
    
    #calculate the accuracy
    accuracy = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy') 
    print("accuracy: %f" %accuracy.mean() + '\n')
    #print accuracy
    
    #calculate the precision
    precision = cross_val_score(clf, train_x, train_y, cv=5, scoring='precision_macro')
    print("precision: %f" %precision.mean() + '\n')
    
    #calculate the recall score
    recall = cross_val_score(clf, train_x, train_y, cv=5, scoring='recall_macro')
    print("recall: %f" %recall.mean() + '\n')
    
    #calculate the f_measure
    f_measure = cross_val_score(clf, train_x, train_y, cv=5, scoring='f1_macro')
    print("f_measure: %f " %f_measure.mean() + '\n')
    
    #generate classification report and MCC and G-mean value
    y_pred = cross_val_predict(clf, train_x, train_y, cv=5)
    G_mean = geometric_mean_score(train_y, y_pred)
    MCC = matthews_corrcoef(train_y, y_pred)
    print("G_mean: %f" %G_mean.mean() + '\n')
    print("MCC: %f" %np.mean(MCC) + '\n')
    
    print("Classification_report:")
    print(metrics.classification_report(train_y, y_pred))

    
def logistic_cross_validation(train_x, train_y):
    np.random.seed(100)
    clf = linear_model.LogisticRegression()
    
    #calculate the accuracy
    accuracy = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy') 
    print("accuracy: %f" %accuracy.mean() + '\n')
    #print accuracy
    
    #calculate the precision
    precision = cross_val_score(clf, train_x, train_y, cv=5, scoring='precision_macro')
    print("precision: %f" %precision.mean() + '\n')
    
    #calculate the recall score
    recall = cross_val_score(clf, train_x, train_y, cv=5, scoring='recall_macro')
    print("recall: %f" %recall.mean() + '\n')
    
    #calculate the f_measure
    f_measure = cross_val_score(clf, train_x, train_y, cv=5, scoring='f1_macro')
    print("f_measure: %f " %f_measure.mean() + '\n' )

    #generate classification report and MCC and G-mean value
    y_pred = cross_val_predict(clf, train_x, train_y, cv=5)
    G_mean = geometric_mean_score(train_y, y_pred)
    MCC = matthews_corrcoef(train_y, y_pred)
    print("G_mean: %f" %G_mean.mean() + '\n')
    print("MCC: %f" %np.mean(MCC) + '\n')
    
    print("Classification_report:")
    print(metrics.classification_report(train_y, y_pred))
    
    
def bayes_cross_validation(train_x, train_y):
    np.random.seed(100)
    clf = GaussianNB()
    
    #calculate the accuracy
    accuracy = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy') 
    print("accuracy: %f" %accuracy.mean() + '\n')
    #print accuracy
    
    #calculate the precision
    precision = cross_val_score(clf, train_x, train_y, cv=5, scoring='precision_macro')
    print("precision: %f" %precision.mean() + '\n')
    
    #calculate the recall score
    recall = cross_val_score(clf, train_x, train_y, cv=5, scoring='recall_macro')
    print("recall: %f" %recall.mean() + '\n')
    
    #calculate the f_measure
    f_measure = cross_val_score(clf, train_x, train_y, cv=5, scoring='f1_macro')
    print("f_measure: %f " %f_measure.mean() + '\n' )

    #generate classification report and MCC and G-mean value
    y_pred = cross_val_predict(clf, train_x, train_y, cv=5)
    G_mean = geometric_mean_score(train_y, y_pred)
    MCC = matthews_corrcoef(train_y, y_pred)
    print("G_mean: %f" %G_mean.mean() + '\n')
    print("MCC: %f" %np.mean(MCC) + '\n')
    
    print("Classification_report:")
    print(metrics.classification_report(train_y, y_pred))
    
    
def randomforest_cross_validation(train_x, train_y):
    np.random.seed(100)
    clf = ensemble.RandomForestClassifier()
    
    #calculate the accuracy
    accuracy = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy') 
    print("accuracy: %f" %accuracy.mean() + '\n')
    #print accuracy
    
    #calculate the precision
    precision = cross_val_score(clf, train_x, train_y, cv=5, scoring='precision_macro')
    print("precision: %f" %precision.mean() + '\n')
    
    #calculate the recall score
    recall = cross_val_score(clf, train_x, train_y, cv=5, scoring='recall_macro')
    print("recall: %f" %recall.mean() + '\n')
    
    #calculate the f_measure
    f_measure = cross_val_score(clf, train_x, train_y, cv=5, scoring='f1_macro')
    print("f_measure: %f " %f_measure.mean() + '\n')

    #generate classification report and MCC and G-mean value
    y_pred = cross_val_predict(clf, train_x, train_y, cv=5)
    G_mean = geometric_mean_score(train_y, y_pred)
    MCC = matthews_corrcoef(train_y, y_pred)
    print("G_mean: %f" %G_mean.mean() + '\n')
    print("MCC: %f" %np.mean(MCC) + '\n')
    
    print("Classification_report:")
    print(metrics.classification_report(train_y, y_pred)) 
    
def nn_cross_validation(train_x, train_y):
    np.random.seed(100)
    clf = MLPClassifier()
    
    #calculate the accuracy
    accuracy = cross_val_score(clf, train_x, train_y, cv=5, scoring='accuracy') 
    print("accuracy: %f" %accuracy.mean() + '\n')
    #print accuracy
    
    #calculate the precision
    precision = cross_val_score(clf, train_x, train_y, cv=5, scoring='precision_macro')
    print("precision: %f" %precision.mean() + '\n')
    
    #calculate the recall score
    recall = cross_val_score(clf, train_x, train_y, cv=5, scoring='recall_macro')
    print("recall: %f" %recall.mean() + '\n')
    
    #calculate the f_measure
    f_measure = cross_val_score(clf, train_x, train_y, cv=5, scoring='f1_macro')
    print("f_measure: %f " %f_measure.mean() + '\n')

    #generate classification report and MCC and G-mean value
    y_pred = cross_val_predict(clf, train_x, train_y, cv=5)
    G_mean = geometric_mean_score(train_y, y_pred)
    MCC = matthews_corrcoef(train_y, y_pred)
    print("G_mean: %f" %G_mean.mean() + '\n')
    print("MCC: %f" %np.mean(MCC) + '\n')
    
    print("Classification_report:")
    print(metrics.classification_report(train_y, y_pred)) 



####################################################################################################################################  
def lr_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = linear_model.LogisticRegression().fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))
    
def knn_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = neighbors.KNeighborsClassifier().fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))
    
def svm_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = SVC(gamma='auto', class_weight='balanced').fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))
 
def rf_baseline(X, Y, X_test, Y_test):
    setup_seed(20)
    clf = ensemble.RandomForestClassifier().fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))

def nn_baseline(X, Y, X_test, Y_test):
    setup_seed(20)
    clf = MLPClassifier(random_state=100).fit(X, Y) 
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))
    
    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
                % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
                % (val_acc, precision, recall, fscore, mcc))

   
 
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1) # Squeeze
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1) # Excitation
        w = torch.sigmoid(w)

        return x * w + b # Scale and add bias

# Network Module
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out)) # [32,64,325,100]

        # Squeeze
        w = F.avg_pool2d(out, [out.size(2),out.size(3)]) #[32,plane,1,1]

        w = F.relu(self.fc1(w))

        w = F.sigmoid(self.fc2(w))

        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.relu = nn.ReLU(100)
        self.linear = nn.Linear(15360, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x)) #[32,64,325,100]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) #[bs,512,41,13]
        out = F.avg_pool2d(out, 4) #[bs,512,10,3]
        out = out.view(out.size(0), -1) #[bs,15360]
        #out = self.relu(out)
        out = self.linear(out)

        return out


def SENet18b():
    return SENet(BasicBlock, [3,4,6,3])

def SENet18():
    return SENet(PreActBlock, [2,2,2,2])    
  
    

  
    










