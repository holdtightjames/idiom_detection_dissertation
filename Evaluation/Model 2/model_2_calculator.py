# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 02:46:17 2022

@author: James
"""

# =============================================================================
# This code checks the metrics for model 2
# =============================================================================

import pandas as pd
import csv
import numpy as np
import json


gold_standard_magpie = []
# Stores the labels to each idiom in the test set

with open('MAGPIE_filtered_split_typebased.jsonl', 'r', encoding='utf-8') as json_file:
    Magpie_corpus = list(json_file)
for line in Magpie_corpus:
    Magpie_corpus_loaded = json.loads(line)
    if Magpie_corpus_loaded['split'] == "test":
        gold_standard_magpie.append(Magpie_corpus_loaded['label'])




model_2_predicition_dataset = pd.read_csv('magpie_output.csv')
idioms_prediction = model_2_predicition_dataset['Prediction']


# Checks the amount of idioms and literals that have been predicted
idiom_amount = 0
literal_amount = 0
for line in idioms_prediction:
    if line == 'i':
        idiom_amount = idiom_amount + 1
    else:
        literal_amount = literal_amount  + 1
print('The amount of idioms:')
print(idiom_amount)
print('The amount of literals:')
print(literal_amount)

# Combines the correct labels and the predicted labels
combined_list = zip(gold_standard_magpie[0:100],idioms_prediction)


# Calculates the overall accuracy much like model 1
def overall_accuracy(comparison_column):
    Tp = 0
    Fn = 0
    for item in comparison_column:
        if item == True:
            Tp = Tp + 1
        elif item == False:
            Fn = Fn + 1
    return (Tp/(len(comparison_column)))


    
with open('magpie_gold_pred_file.tsv','wt', encoding="utf-8") as out_file:
    headersList =['GoldStandard','Predicted']
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(headersList)
    for gold,pred in combined_list:
        tsv_writer.writerow([gold, pred])
        
        



idiom_output = pd.read_csv('magpie_gold_pred_file.tsv', sep='\t')
gold_standard_idiom = idiom_output ["GoldStandard"]
my_idiom_predicition = idiom_output ["Predicted"]



compare_list = zip(gold_standard_idiom,my_idiom_predicition)
comparison_column = np.where(idiom_output["GoldStandard"] == idiom_output["Predicted"], True, False)
accuracy = overall_accuracy(comparison_column)



# Calculates the metrics of model 2 and the performance
def calc_confusion_matrix():
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for gold,predicted in compare_list:
        if (gold == "i" and predicted == "i"):
            TP = TP + 1
        elif (gold == "l" and predicted == "l"):
            TN = TN + 1
        elif (gold == "l" and predicted == "i"):
            FP = FP + 1
        elif (gold == "i" and predicted == "l"):
            FN = FN + 1
            
    recall = TP/(TP + FN)
    precision = TP/(TP + FP)
    f1_score = (2*precision*recall)/(recall + precision)
    print('recall is ' + str(recall))
    print('precision is ' + str(precision))
    print('f1_score is ' + str(f1_score))


calc_confusion_matrix()
print('The accuracy is '+ str(accuracy))
