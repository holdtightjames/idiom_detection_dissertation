# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:24:02 2022

@author: James
"""

import json
import torch
import math
import numpy as np
from transformers import BertTokenizer, BertModel
import sys, re, getopt, csv, nltk
import pandas as pd
import random
from nltk.corpus import wordnet
from scipy.spatial.distance import cosine

gold_standard_magpie = []
masked_idiom_list = []
masked_idiom_list_spaces = []
predicted_sentences = []
gold_standard_magpie_shorter = []
predicted_sentences_flattened = []
idiom_list = []
cosine_sim_values = []
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def synonym_antonym_extractor(phrase):
    synonyms = []
    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            synonyms.append(l.name())

    return set(synonyms)
     # print(set(synonyms))

# output = synonym_antonym_extractor(phrase="good")
# highest_syn = list(output).pop()


def noun_compound_vector_sentence(text):

    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True, # Whether the model returns all hidden-states.
                                      )
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding[:5]



file1 = open('Static_idioms.txt', 'r')
for line in file1:
    idiom_list.append(line.strip())


for idiom in idiom_list[:100]:
    idiom = idiom.split()
    replacement_before = synonym_antonym_extractor (idiom[0])
    if len(replacement_before) > 0:
        replacement = list(replacement_before).pop()
    else:
        replacement = " "
    # print(idiom)
    # idiom[0] = replacement
    idiom[(len(idiom)-1)] = replacement
#     # print(len(idiom))
    # idiom[math.floor((len(idiom)/2))] = replacement
    masked_idiom_list.append(idiom)

for item in masked_idiom_list:
    one_sentence = ' '.join(item)
    masked_idiom_list_spaces.append(one_sentence)
# print(masked_idiom_list_spaces)
    
both_list = zip(idiom_list[:100],masked_idiom_list_spaces)
# smaller_masked_list = masked_idiom_list_spaces[:2000]
# gold_standard_magpie_shorter = gold_standard_magpie[:2000]


for org,new in both_list:
    original_sentence_embedding = noun_compound_vector_sentence(org)
    new_sentence_embedding = noun_compound_vector_sentence(new)
    cosine_sim = 1 - cosine(original_sentence_embedding, new_sentence_embedding)
    cosine_sim_values.append(cosine_sim)
    

combined_list = zip(idiom_list[:100],cosine_sim_values)

# print(predicted_sentences_flattened)


with open('end_syn_idiom_replaced.csv','wt', encoding="utf-8") as out_file:
    headersList =['Original Idiom','Cosine Sim']
    tsv_writer = csv.writer(out_file)
    tsv_writer.writerow(headersList)
    for Original,Cosine in combined_list:
        tsv_writer.writerow([Original, Cosine])
        
        
        