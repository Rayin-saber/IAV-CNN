# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:31:29 2019

@author: yinr0002
"""

import os, sys
import pandas as pd
import numpy as np
import random
import torch
import warnings
import math

#sys.path.append(os.path.abspath("/content/drive/My Drive/Colab Notebooks/bioinformatics/code"))
sys.path.append(os.path.abspath("C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/5_Journal/Bioinformatics_2/code/"))

from Bio import SeqIO
from model import fasta_to_csv
from model import calculate_label
from model import generate_feature
from model import strain_selection
from model import replace_uncertain_amino_acids
from model import train_test_split_data

warnings.filterwarnings('ignore')
#os.chdir('/content/drive/My Drive/Colab Notebooks/bioinformatics/data')
os.chdir('C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/5_Journal/Bioinformatics_2/data/')


####convert fasta file to csv for raw sequence data
##H1N1_seq = SeqIO.parse('sequence/H1N1_sequence.fa', 'fasta')
##fasta_to_csv(H1N1_seq)
#H5N1_seq = SeqIO.parse('sequence/H5N1/H5N1_sequence_HA1.fas', 'fasta')
#fasta_to_csv(H5N1_seq)

H1N1_Antigenic_dist = pd.read_csv('antigenic/H1N1_antigenic.csv')
H1N1_seq = pd.read_csv('sequence/H1N1/H1N1_sequence_HA1.csv', names=['seq', 'description'])
H3N2_Antigenic_dist = pd.read_csv('antigenic/H3N2_antigenic.csv')
H3N2_seq = pd.read_csv('sequence/H3N2/H3N2_sequence_HA1.csv', names=['seq', 'description'])
H5N1_Antigenic_dist = pd.read_csv('antigenic/H5N1_antigenic.csv')
H5N1_seq = pd.read_csv('sequence/H5N1/H5N1_sequence_HA1.csv', names=['seq', 'description'])
#
##divide residue sites based on five epitope regions
#epitope_a = [118, 120, 121, 122, 126, 127, 128, 129, 132, 133, 134, 135, 137, 139, 140, 141, 142, 143, 146, 147, 149, 165, 252, 253]
#epitope_b = [124, 125, 152, 153, 154, 155, 156, 157, 160, 162, 163, 183, 184, 185, 186, 187, 189, 190, 191, 193, 194, 196]
#epitope_c = [34, 35, 36, 37, 38, 40, 41, 43, 44, 45, 269, 270, 271, 272, 273, 274, 276, 277, 278, 283, 288, 292, 295, 297, 298, 302, 303, 305, 306, 307, 308, 309, 310]
#epitope_d = [89, 94, 95, 96, 113, 117, 163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174, 176, 179, 198, 200, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 222, 223, 224, 225, 226, 227, 235, 237, 239, 241, 243, 244, 245]
#epitope_e = [47, 48, 50, 51, 53, 54, 56, 57, 58, 66, 68, 69, 70, 71, 72, 73, 74, 75, 78, 79, 80, 82, 83, 84, 85, 86, 102, 257, 258, 259, 260, 261, 263, 267]
#epitopes = {'epitope_a': epitope_a, 'epitope_b': epitope_b, 'epitope_c': epitope_c, 'epitope_d': epitope_d, 'epitope_e':epitope_e}
#

#William Lees's work (A computational analysis of the antigenic properties of haemagglutinin in influenza A H3N2)
#divide residue sites based on proposed five epitope regions
#H1N1_new_epitope_a = [62, 63, 64, 91, 118, 120, 121, 122, 123, 126, 127, 128, 129, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 
#                 143, 145, 146, 147, 148, 149, 165, 252, 253]
#H1N1_new_epitope_b = [124, 125, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 183, 184, 185, 186, 187, 189, 190, 191, 193,
#                 194, 196]
#H1N1_new_epitope_c = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 269, 270, 271, 272, 273, 274, 276, 277, 278, 280, 
#                 282, 283, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 305, 306, 
#                 307, 308, 309, 310]
#H1N1_new_epitope_d = [88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 100, 111, 113, 117, 163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174,
#                 175, 176, 177, 179, 180, 181, 197, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 
#                 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 235, 236, 237, 239, 
#                 240, 241, 242, 243, 244, 245, 254, 255]
#H1N1_new_epitope_e = [47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
#                 79, 80, 81, 82, 83, 84, 85, 86, 102, 103, 104, 105, 106, 107, 108, 112, 256, 257, 258, 259, 260, 261, 262, 263, 265,
#                 266, 267, 268]
#H1N1_new_epitopes = {'new_epitope_a': H1N1_new_epitope_a, 'new_epitope_b': H1N1_new_epitope_b, 'new_epitope_c': H1N1_new_epitope_c, 
#                'new_epitope_d': H1N1_new_epitope_d, 'new_epitope_e': H1N1_new_epitope_e}
#H1N1_epitope_data = generate_feature(H1N1_new_epitopes, H1N1_Antigenic_dist, H1N1_seq)
#H1N1_epitope_data.to_csv('training/William Lees/H1N1_new_epitope_data.csv')
#
#
#H3N2_new_epitope_a = [71, 72, 98, 122, 124, 126, 127, 130, 131, 132, 133, 135, 137, 138, 140, 141, 142, 143, 144, 145, 146, 148, 149,
#                      150, 151, 152, 168, 255]
#H3N2_new_epitope_b = [128, 129, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 
#                      197, 198, 199]
#H3N2_new_epitope_c = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 271, 272, 273, 274, 275, 276, 278, 279, 280, 282, 
#                      284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 307, 
#                      308, 309, 310, 311, 312, 313, 314]
#H3N2_new_epitope_d = [95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 107, 117, 118, 120, 121, 166, 167, 169, 170, 171, 172, 173, 174, 
#                      175, 176, 177, 178, 179, 180, 182, 183, 184, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 
#                      214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 
#                      236, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248, 257, 258]
#H3N2_new_epitope_e = [56, 57, 58, 59, 60, 62, 63, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
#                      88, 89, 90, 91, 92, 93, 94, 109, 110, 111, 112, 113, 114, 115, 119, 259, 260, 261, 262, 263, 264, 265, 267, 268,
#                      268, 270]
#H3N2_new_epitopes = {'new_epitope_a': H3N2_new_epitope_a, 'new_epitope_b': H3N2_new_epitope_b, 'new_epitope_c': H3N2_new_epitope_c, 
#                'new_epitope_d': H3N2_new_epitope_d, 'new_epitope_e': H3N2_new_epitope_e}
##H3N2_epitope_data = generate_feature(H3N2_new_epitopes, H3N2_Antigenic_dist, H3N2_seq)
##H3N2_epitope_data.to_csv('training/William Lees/H3N2_new_epitope_data.csv')
#
#
##0 is H1N1, 1 is H3N2, 2 is H5N1, map antigenic site across different subtypes
#embedding = pd.read_csv('sequence/embedding.csv', header=None)
#def map_antigenic_site(embedding_table, input_site, subtype):
#    input_index = []
#    output_index = []
#    if subtype == 1:
#        input_index = list(embedding_table[8])
#    for i in range(0, len(input_site)):
#        for j in range(embedding_table.shape[0]):
#            result = []
#            if input_index[j] == str(input_site[i]):
#                result = int(embedding_table[12][j])
#                break                
#        output_index.append(result)
#    return output_index
#
#H5N1_new_epitope_a = map_antigenic_site(embedding, H3N2_new_epitope_a, 1)
#H5N1_new_epitope_b = map_antigenic_site(embedding, H3N2_new_epitope_b, 1)
#H5N1_new_epitope_c = map_antigenic_site(embedding, H3N2_new_epitope_c, 1)
#H5N1_new_epitope_d = map_antigenic_site(embedding, H3N2_new_epitope_d, 1)
#H5N1_new_epitope_e = map_antigenic_site(embedding, H3N2_new_epitope_e, 1)
#H5N1_new_epitopes = {'new_epitope_a': H5N1_new_epitope_a, 'new_epitope_b': H5N1_new_epitope_b, 'new_epitope_c': H5N1_new_epitope_c, 
#                'new_epitope_d': H5N1_new_epitope_d, 'new_epitope_e': H5N1_new_epitope_e}
##H5N1_epitope_data = generate_feature(H5N1_new_epitopes, H5N1_Antigenic_dist, H5N1_seq)
##H5N1_epitope_data.to_csv('training/William Lees/H5N1_new_epitope_data.csv')




#Peng Yousong's work
#divide residue sites based on ten regional bands and generate training data with new features
#H1N1 
H1N1_regional_1 = [125,126,127,152,153,154,155,156,157,158,190,191,193]
H1N1_regional_2 = [119,120,121,122,123,124,128,129,130,131,132,133,150,159,160,161,162,184,185,186,187,189,194,195,196,198,243,245]
H1N1_regional_3 = [89,92,93,94,114,115,116,118,134,137,138,139,140,141,142,143,144,146,163,164,165,181,183,200,202,208,209,210,211,
              212,213,214,215,216,217,219,220,221,222,223,224,226,228,230,231,241,252,253]
H1N1_regional_4 = [54,56,60,66,68,69,70,71,72,73,74,85,86,87,95,96,97,98,99,100,102,112,113,166,168,169,171,172,203,204,205,206,207,
              218,232,233,235,236,237,238,239]
H1N1_regional_5 = [47,48,49,50,51,53,75,77,81,82,83,84,103,104,106,107,109,110,170,258,259,260,261,263,264,265,267,269]
H1N1_regional_6 = [39,40,43,44,45,46,262,270,271,272,273,274,275,276,277,278,282,283,296,297,298,299,300,301,302]
H1N1_regional_7 = [34,35,36,37,38,287,288,289,291,292,294,295,303,304,305,306,307,308,309]
H1N1_regional_8 = [30,31,32,290,310,311,312,313]
H1N1_regional_9 = [12,13,14,15,17,18,19,20,21,22,23,24,25,28,29,314,315,316,317]
H1N1_regional_10 = [1,2,3,4,5,6,7,8,9,10,11,318,319,320,321,322,323,324]
H1N1_regional_band = {'regional_1':H1N1_regional_1, 'regional_2':H1N1_regional_2, 'regional_3':H1N1_regional_3, 'regional_4':H1N1_regional_4,
                      'regional_5':H1N1_regional_5, 'regional_6':H1N1_regional_6, 'regional_7':H1N1_regional_7, 'regional_8':H1N1_regional_8, 
                      'regional_9':H1N1_regional_9, 'regional_10':H1N1_regional_10}

#H1N1_regional_data = generate_feature(H1N1_regional_band, H1N1_Antigenic_dist, H1N1_seq)
#H1N1_regional_data.to_csv('training/Peng Yousong/H1N1_regional_data.csv')

#H3N2 
H3N2_regional_1 = [129,130,131,155,156,157,158,159,160,196]
H3N2_regional_2 = [126,127,128,132,133,134,135,136,153,162,163,165,188,189,190,192,193,194,197,198,199,201,248]
H3N2_regional_3 = [74,100,101,122,123,124,125,137,138,140,141,142,143,144,145,147,149,150,166,167,168,186,187,
                   211,212,213,214,215,216,217,219,222,223,224,225,226,227,233,234,244,255,257]
H3N2_regional_4 = [63,65,75,77,78,79,80,81,82,93,94,95,96,102,103,104,105,106,109,119,120,121,169,171,172,174,
                   175,207,208,209,210,236,238,239,240,242,259,260]
H3N2_regional_5 = [57,58,59,60,62,83,85,89,90,91,92,110,114,173,261,262,263,264,267,269,271]
H3N2_regional_6 = [49,50,53,54,55,56,272,273,274,275,276,277,278,279,280,284,285,298,299,300,301]
H3N2_regional_7 = [44,45,46,47,48,289,290,291,292,293,296,297,307,308,310,311,312]
H3N2_regional_8 = [40,41,313,315]
H3N2_regional_9 = [22,23,24,25,27,29,31,32,33,34,35,37,38,39,318]
H3N2_regional_10 = [1,2,3,4,5,6,7,8,9,10,12,14,18,20,21,321,323,324,325,326,327,328]
H3N2_regional_band = {'regional_1':H3N2_regional_1, 'regional_2':H3N2_regional_2, 'regional_3':H3N2_regional_3, 'regional_4':H3N2_regional_4,
                      'regional_5':H3N2_regional_5, 'regional_6':H3N2_regional_6, 'regional_7':H3N2_regional_7, 'regional_8':H3N2_regional_8, 
                      'regional_9':H3N2_regional_9, 'regional_10':H3N2_regional_10}

#H3N2_regional_data = generate_feature(H3N2_regional_band, H3N2_Antigenic_dist, H3N2_seq)
#H3N2_regional_data.to_csv('training/Peng Yousong/H3N2_regional_data.csv')

#H5N1
H5N1_regional_1 = [124,125,126,151,152,153,154,155,156,192]
H5N1_regional_2 = [119,120,121,122,123,127,128,129,131,149,158,159,161,184,185,186,188,189,190,193,194,195,242,244]
H5N1_regional_3 = [89,90,93,115,116,118,132,133,134,136,137,138,139,140,141,142,143,145,162,163,164,182,183,197,207,
                   208,209,210,211,212,213,215,218,220,221,222,223,227,229,240,251,252]
H5N1_regional_4 = [54,66,68,69,71,72,73,85,86,87,88,94,95,96,97,98,99,111,112,113,114,165,167,168,169,170,171,203,
                   204,205,206,217,219,230,231,232,234,235,236,237,238,255,256]
H5N1_regional_5 = [48,50,51,53,74,75,82,83,84,102,103,104,106,107,109,110,257,258,259,260,262,263,264,266,268,270,271]
H5N1_regional_6 = [39,40,43,44,45,46,47,261,269,272,273,274,275,276,277,281,282,295,296,297,298,299,300]
H5N1_regional_7 = [34,35,36,37,38,286,287,288,290,293,294,301,302,303,304,305,307]
H5N1_regional_8 = [30,31,289,309,310,312]
H5N1_regional_9 = [12,13,14,15,17,20,21,22,23,25,28,29,315]
H5N1_regional_10 = [1,2,8,10,11,318,319,320]
H5N1_regional_band = {'regional_1':H5N1_regional_1, 'regional_2':H5N1_regional_2, 'regional_3':H5N1_regional_3, 'regional_4':H5N1_regional_4,
                      'regional_5':H5N1_regional_5, 'regional_6':H5N1_regional_6, 'regional_7':H5N1_regional_7, 'regional_8':H5N1_regional_8, 
                      'regional_9':H5N1_regional_9, 'regional_10':H5N1_regional_10}

#H5N1_regional_data = generate_feature(H5N1_regional_band, H5N1_Antigenic_dist, H5N1_seq)
#H5N1_regional_data.to_csv('training/Peng Yousong/H5N1_regional_data.csv')
################################################################################################################################


####################################################################################################################################
#Yu-Chieh Liao method (Bioinformatics models for predicting antigenic variants of influenza A/H3N2 virus)
group1 = ['A', 'I', 'L', 'M', 'P', 'V']
group2 = ['F', 'W', 'Y']       
group3 = ['N', 'Q', 'S', 'T']
group4 = ['D', 'E', 'H', 'K', 'R']
group5 = ['C']
group6 = ['G']

def Liao_feature_engineering(distance_input, seq_input, subtype):
    distance_label = calculate_label(distance_input)
    label = {'label': distance_label}
    label = pd.DataFrame(label)
    
    index = pd.Series(np.arange(distance_input.shape[0]))
    length = len(seq_input['seq'].iloc[0])
    if subtype == 0:
        columns = list(range(1, 328, 1))
    elif subtype == 1:
        columns = list(range(1, 330, 1))
    elif subtype == 2:
        columns = list(range(1, 321, 1))
    for col in range(len(columns)):
        columns[col] = str(columns[col])
    Mut_feature = pd.DataFrame(index=index, columns=columns)
    
    for i in range(0, distance_input.shape[0]):
        strain_1 = []
        strain_2 = []
        for j in range(0, seq_input.shape[0]):
            if seq_input['description'].iloc[j].upper() == distance_input['Strain1'].iloc[i].upper():
                strain_1 = seq_input['seq'].iloc[j].upper()
            if seq_input['description'].iloc[j].upper() == distance_input['Strain2'].iloc[i].upper():
                strain_2 = seq_input['seq'].iloc[j].upper()
        for a in range(0, length):
            if strain_1[a] in group1 and strain_2[a] in group1:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group2 and strain_2[a] in group2:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group3 and strain_2[a] in group3:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group4 and strain_2[a] in group4:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group5 and strain_2[a] in group5:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group6 and strain_2[a] in group6:
                Mut_feature.iloc[i][a] = 0
            else:
               Mut_feature.iloc[i][a] = 1
    Mut_feature = Mut_feature.join(label)   
    return(Mut_feature)
    
#Liao_feature_H1N1 = Liao_feature_engineering(H1N1_Antigenic_dist, H1N1_seq, 0)
#Liao_feature_H1N1.to_csv('training/Yu-Chieh Liao/H1N1.csv')
#Liao_feature_H3N2 = Liao_feature_engineering(H3N2_Antigenic_dist, H3N2_seq, 1)
#Liao_feature_H3N2.to_csv('training/Yu-Chieh Liao/H3N2.csv')
#Liao_feature_H5N1 = Liao_feature_engineering(H5N1_Antigenic_dist, H5N1_seq, 2)
#Liao_feature_H5N1.to_csv('training/Yu-Chieh Liao/H5N1.csv')
###########################################################################################################################
    
###########################################################################################################################
##Yuhua Yao method (Predicting influenza antigenicity from Hemagglutintin sequence data based on a joint random forest method)
Yao_embedding = pd.read_csv('sequence/NIEK910102_matrix.csv')
Yao_pair_aa = []
Yao_pair_aa_value = [] 
for i in range(1, Yao_embedding.shape[0]): #start from index 1
    aa_pair = 0
    aa_value = 0
    for j in range(1, Yao_embedding.shape[1]):
        aa_pair = Yao_embedding.iloc[i][0] + Yao_embedding.iloc[0][j]
        aa_value = Yao_embedding.iloc[i][j]
        if not math.isnan(float(aa_value)):
            Yao_pair_aa.append(aa_pair)
            Yao_pair_aa_value.append(aa_value)
Yao_embedding_table = pd.DataFrame({'aa_pair': Yao_pair_aa, 'aa_value': Yao_pair_aa_value})


def Yao_feature_engineering(distance_input, seq_input, subtype):
    distance_label = calculate_label(distance_input)
    label = {'label': distance_label}
    label = pd.DataFrame(label)
    
    index = pd.Series(np.arange(distance_input.shape[0]))
    length = len(seq_input['seq'].iloc[0])    
    if subtype == 0:
        columns = list(range(1, 328, 1))
    elif subtype == 1:
        columns = list(range(1, 330, 1))
    elif subtype == 2:
        columns = list(range(1, 321, 1))
    for col in range(len(columns)):
        columns[col] = str(columns[col])
    Mut_feature = pd.DataFrame(index=index, columns=columns)

    for i in range(0, distance_input.shape[0]):
        strain_1 = []
        strain_2 = []
        for j in range(0, seq_input.shape[0]):
            if seq_input['description'].iloc[j].upper() == distance_input['Strain1'].iloc[i].upper():
                strain_1 = seq_input['seq'].iloc[j].upper()
            if seq_input['description'].iloc[j].upper() == distance_input['Strain2'].iloc[i].upper():
                strain_2 = seq_input['seq'].iloc[j].upper()
        for a in range(0, length):
            aa_pair_1 = strain_1[a] + strain_2[a]
            aa_pair_2 = strain_2[a] + strain_1[a]
            for b in range(0, Yao_embedding_table.shape[0]):
                if strain_1[a] == '-' or strain_2[a] == '-':
                    Mut_feature.iloc[i][a] = 0
                    break
                elif aa_pair_1 == Yao_embedding_table['aa_pair'].iloc[b] or aa_pair_2 == Yao_embedding_table['aa_pair'].iloc[b]:
                    Mut_feature.iloc[i][a] = Yao_embedding_table['aa_value'].iloc[b]
                    break

    Mut_feature = Mut_feature.join(label)
    return(Mut_feature)

#H1N1_seq = replace_uncertain_amino_acids(H1N1_seq)
#Yao_feature_H1N1 = Yao_feature_engineering(H1N1_Antigenic_dist, H1N1_seq, 0)
#Yao_feature_H1N1.to_csv('training/Yuhua Yao/H1N1.csv')
#
#H3N2_seq = replace_uncertain_amino_acids(H3N2_seq)
#Yao_feature_H3N2 = Yao_feature_engineering(H3N2_Antigenic_dist, H3N2_seq, 1)
#Yao_feature_H3N2.to_csv('training/Yuhua Yao/H3N2.csv')

#H5N1_seq = replace_uncertain_amino_acids(H5N1_seq)
#Yao_feature_H5N1 = Yao_feature_engineering(H5N1_Antigenic_dist, H5N1_seq, 2)
#Yao_feature_H5N1.to_csv('training/Yuhua Yao/H5N1.csv')

###################################################################################################################################
def cnn_training_data(Antigenic_dist, seq):
    raw_data = strain_selection(Antigenic_dist, seq)
    #replace unambiguous with substitutions
    Btworandom = 'DN'
    Jtworandom = 'IL'
    Ztworandom = 'EQ'
    Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(0, 2):
        for j in range(0, len(raw_data[0])):
            seq = raw_data[i][j]
            seq = seq.replace('B', random.choice(Btworandom))
            seq = seq.replace('J', random.choice(Jtworandom))
            seq = seq.replace('Z', random.choice(Ztworandom))
            seq = seq.replace('X', random.choice(Xallrandom))
            raw_data[i][j] = seq
            
    #embedding with ProVect    
    df = pd.read_csv('protVec_100d_3grams.csv', delimiter = '\t')
    trigrams = list(df['words'])
    trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
    trigram_vecs = df.loc[:, df.columns != 'words'].values

    feature = []
    label = raw_data[2]
    for i in range(0, len(raw_data[0])):
        trigram1 = []
        trigram2 = []
        strain_embedding = []
        seq1 = raw_data[0][i]
        seq2 = raw_data[1][i]
    
        for j in range(0, len(raw_data[0][0])-2):
            trigram1 = seq1[j:j+3]
            if trigram1[0] == '-' or trigram1[1] == '-' or trigram1[2] == '-':
                tri1_embedding = trigram_vecs[trigram_to_idx['<unk>']]
            else:
                tri1_embedding = trigram_vecs[trigram_to_idx[trigram1]]
        
            trigram2 = seq2[j:j+3]
            if trigram2[0] == '-' or trigram2[1] == '-' or trigram2[2] == '-':
                tri2_embedding = trigram_vecs[trigram_to_idx['<unk>']]
            else:
                tri2_embedding = trigram_vecs[trigram_to_idx[trigram2]]
        
            tri_embedding = tri1_embedding - tri2_embedding
            strain_embedding.append(tri_embedding)
  
        feature.append(strain_embedding)
    return feature, label
###########################################################################################################################


































