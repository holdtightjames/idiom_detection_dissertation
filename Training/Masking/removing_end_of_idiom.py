# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:24:02 2022

@author: James
"""


# =============================================================================
# Masking Idioms at different parts and testing BERT's ability to predict the idiom.
# =============================================================================

import math
import numpy as np
from transformers import AutoTokenizer
from transformers import TFAutoModelForMaskedLM
import csv
import pandas as pd

masked_idiom_list = []
masked_idiom_list_spaces = []
predicted_sentences = []
predicted_sentences_flattened = []
idiom_list = []


model_checkpoint = "distilbert-base-uncased"
model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


#Function to check how many idioms were predicted correctly
def overall_accuracy(comparison_column):
    Tp = 0
    Fn = 0
    for item in comparison_column:
        if item == True:
            Tp = Tp + 1
        elif item == False:
            Fn = Fn + 1
    return (Tp/(len(comparison_column)))


# Read in the static_idioms file
file1 = open('Static_idioms.txt', 'r')
for line in file1:
    idiom_list.append(line.strip())



# Masking idioms in the list at the points specified
for idiom in idiom_list:
    replacement = '[MASK]'
    idiom = idiom.split()
    # idiom[0] = replacement #Masks the begining of the idiom
    idiom[(len(idiom)-1)] = replacement #Masks the idiom at the end
    # idiom[math.floor((len(idiom)/2))] = replacement #Masks randomly in the middle of the idiom
    masked_idiom_list.append(idiom)

#Add spaces back into the sentence with the mask now added
for item in masked_idiom_list:
    one_sentence = ' '.join(item)
    masked_idiom_list_spaces.append(one_sentence)
    


#Code to see if the BERT model can predict the idiom
#Code inspired from https://huggingface.co/course/chapter7/3?fw=tf
for idiom in masked_idiom_list_spaces:
    text = idiom
    inputs = tokenizer(text, return_tensors="np")
    token_logits = model(**inputs).logits
    mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = np.argsort(-mask_token_logits)[:1].tolist()
    for token in top_5_tokens:
        predicted_sentences.append({text.replace(tokenizer.mask_token, tokenizer.decode([token]))})
        #Only using the top predicted sentence rather than the top 5.
        
#Creates a list of all the predicted sentences in the same strucuture as static idioms
for item in predicted_sentences:
    str_val = " ".join(item)
    predicted_sentences_flattened.append(str_val)


#Creating a new list that is the gold standard idiom and the one predicted
combined_list = zip(idiom_list,predicted_sentences_flattened)


#Writing to a file that contains the gold standard and the predicted one
with open('end_idiom_replaced.tsv','wt', encoding="utf-8") as out_file:
    headersList =['GoldStandard','Predicted']
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(headersList)
    for gold_idiom,predicted_idiom in combined_list:
        tsv_writer.writerow([gold_idiom, predicted_idiom])
        
        
#Compares the idiom predicted to the gold standard
idiom_output = pd.read_csv('end_idiom_replaced.tsv', sep='\t')
comparison_column = np.where(idiom_output["GoldStandard"] == idiom_output["Predicted"], True, False)
accuracy = overall_accuracy(comparison_column)
print(accuracy)

# 0.21727019498607242 accuracy for middle
# 0.12813370473537605 accuracy for begining
# 0.09749303621169916 accuracy for end

