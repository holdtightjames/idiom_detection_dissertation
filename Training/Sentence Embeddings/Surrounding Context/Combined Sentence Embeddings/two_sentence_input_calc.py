# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 06:21:31 2022

@author: James
"""
# =============================================================================
# This is a calculator that generates a mean for all the feature values in the sentence embedding
# =============================================================================


import pandas as pd
import csv

# This is changed to output for both the (before and target) and (after and target)
noun_compound_dataset = pd.read_csv('two_sentence_input_embeddings_after_target.csv')
sentence_vectors = noun_compound_dataset["Sentence Vector"]

vector_dict = {}


# Sums all the feature values in the embedding and averages it over 5
i = 0;
for item in sentence_vectors:
    i = i + 1
    begining_string = item.find('[')
    end_of_string = item.find(']')
    new_word = item[begining_string + 1:end_of_string]
    all_string_item = new_word.split(',')
    for item2 in all_string_item:
        if i in vector_dict:   
            vector_dict[i] = float(item2) + vector_dict.get(i)
        else:
            vector_dict[i] = float(item2)
    vector_dict[i] = vector_dict.get(i)/5
    


   
# Outputs the average into a csv file
with open('two_sentence_input_embeddings_target_after_output.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerows(vector_dict.items())
