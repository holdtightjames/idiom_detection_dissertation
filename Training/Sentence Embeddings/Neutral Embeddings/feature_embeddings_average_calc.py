# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 06:21:31 2022

@author: James
"""

import pandas as pd
import csv

# =============================================================================
# Code to get the mean of feature vectors in the neutral embeddings
# The neutral embeddings generated from the P1_sents dataset
# =============================================================================

# Retrieves the TSV file containing all the sentence embeddings 
sentence_embeddings_output = pd.read_csv('neutral_sentence_embeddings.tsv', sep='\t')
context_sentence_embeddings = sentence_embeddings_output["Sentence Vector"]


# This will sum all the feature vectors in the embedding and divide it by the total features inside (5)
# This generates the mean features value within the embedding
vector_dict = {}
i = 0;
for item in context_sentence_embeddings:
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



with open('neutral_features_mean.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerows(vector_dict.items())