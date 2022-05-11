# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 03:08:36 2022

@author: James
"""


# =============================================================================
# Word embeddings between NCs
# =============================================================================



from scipy.spatial.distance import cosine
import torch
from transformers import BertTokenizer, BertModel
import csv
import pandas as pd



df_nc = pd.read_csv('reddy2011.csv', usecols=[0,1], header=None)
nc_list = []

nc_head = df_nc[0]
#Variable to store all the heads of the NCs
nc_mod = df_nc[1]
#Variable to store all the modifiers of the NCs
combined_list = []
#List that contains the head and modifier of each NC
NC_diff_list = []
# Variable that stores the cosine sim between the head and modifier
finished_list = []
# Stores the NC and the cosine sim between the head and modifier

nc_list = zip(list(nc_head),list(nc_mod))

# Merges the two lists together
for head,mod in nc_list:
    combined_list.append(head + ' ' + mod)
    


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer that has a large vocabulary for all words. where there is no words, it combines the words it knows

# Based on work from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/ code
# Returns all the word embeddings of the NC; and all features of the embedding (768)
def noun_compound_vector(text):
    token_vecs_sum_gen = []
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True,
                                      )
    with torch.no_grad():
    
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)
    token_vecs = hidden_states[-2][0]
    
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum_gen.append(sum_vec)
        # Sums the last 4 layers of BERT
    return token_vecs_sum_gen


# Calculates the cosine sim between the mod and head embeddings
for item in combined_list:
    token_vecs_sum = 0
    token_vecs_sum = noun_compound_vector(item)
    print('The word: ' + item)
    NC_diff = 1 - cosine(token_vecs_sum[1], token_vecs_sum[2])
    print('The NC difference ')
    NC_diff_list.append(NC_diff)
    
    


finished_list = zip(combined_list,NC_diff_list)
with open('nc_mod_head_cosine.csv','wt') as out_file:
        headersList =['NC','Cosine Sim']
        tsv_writer = csv.writer(out_file)
        tsv_writer.writerow(headersList)
        for NC,vector_val in finished_list:
            tsv_writer.writerow([NC, vector_val])