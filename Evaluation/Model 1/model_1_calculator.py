# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 02:46:17 2022

@author: James
"""

# =============================================================================
# This file calculates all the metrics for model 1's output
# =============================================================================

import pandas as pd
import csv
import numpy as np


wiktionary_dataset = pd.read_csv('wiktionary_dataset.csv')
idioms_gold = wiktionary_dataset['CLASSIFICATION']

wiktionary_predicition_dataset = pd.read_csv('wiktionary_output.csv')
idioms_prediction = wiktionary_predicition_dataset['Prediction']


combined_list = zip(idioms_gold[:100],idioms_prediction)
# Combines the gold standard idioms and the ones predicted by me


# Checks if the values are fully equal or not and then generates an overall accuracy
def overall_accuracy(comparison_column):
    Tp = 0
    Fn = 0
    for item in comparison_column:
        if item == True:
            Tp = Tp + 1
        elif item == False:
            Fn = Fn + 1
    return (Tp/(len(comparison_column)))


# Creates a new file that contains the gold standard idioms from the wiktionary dataset and the ones predicted from model 1
with open('wiktionary_gold_pred_file.tsv','wt', encoding="utf-8") as out_file:
    headersList =['GoldStandard','Predicted']
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(headersList)
    for gold,pred in combined_list:
        tsv_writer.writerow([gold, pred])
        
    
idiom_output = pd.read_csv('wiktionary_gold_pred_file.tsv', sep='\t')
gold_standard_idiom = idiom_output ["GoldStandard"]
my_idiom_predicition = idiom_output ["Predicted"]

compare_list = zip(gold_standard_idiom,my_idiom_predicition)
comparison_column = np.where(idiom_output["GoldStandard"] == idiom_output["Predicted"], True, False)
# Creates a list where each instance of an idiom that is accurately predicted is labelled True and vice versa
accuracy = overall_accuracy(comparison_column)




def calc_confusion_matrix():
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for gold,predicted in compare_list:
        if (gold == "idiom" and predicted == "idiom"):
            TP = TP + 1
        elif (gold == "literal" and predicted == "literal"):
            TN = TN + 1
        elif (gold == "literal" and predicted == "idiom"):
            FP = FP + 1
        elif (gold == "idiom" and predicted == "literal"):
            FN = FN + 1
            
    recall = TP/(TP + FN)
    precision = TP/(TP + FP)
    f1_score = (2*precision*recall)/(recall + precision)
    # Generates the metrics for all the data
    print('recall is ' + str(recall))
    print('precision is ' + str(precision))
    print('f1_score is ' + str(f1_score))


calc_confusion_matrix()
print('The accuracy is '+ str(accuracy))












