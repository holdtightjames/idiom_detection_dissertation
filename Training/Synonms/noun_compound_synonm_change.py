# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 03:08:36 2022

@author: James
"""

# =============================================================================
# Checks the cosine similarity between the sentence embeddings of NC's and their synonm counterpart
# =============================================================================

import csv
import pandas as pd
from scipy.spatial.distance import cosine
import torch
from transformers import BertTokenizer, BertModel


nc_orignal_sentence_vector = []
#List to hold all the orignal NCs sentence embeddings

syn_orignal_sentence_vector = []
#List to hold all the synonm sentence embeddings

nc_sn_list = []
# List to hold all the synonms of the NC


combined_sentence_embeddings = []
# A list that has the sentence embeddings for the orignal NC and the synonm

all_vector_simi = []
# A list containing all the cosine similarities between each embedding

vector_embeddings_with_NC = []
# List with the NC and its cosine sim value



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



data_en_dataset = pd.read_csv('data_en.tsv', sep='\t')
nc_pd = data_en_dataset['compound']
nc_sn_pd = data_en_dataset['Synonyms']


# Retrieves the first synonm of the NC
for item in nc_sn_pd:
    split_items = item.split(';');
    nc_sn_list.append(split_items[0])


# Combines the NC and its synonm
combined_list = zip(nc_pd,nc_sn_list)



# Code inspired from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
def sentence_embeddings(text):
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
    # Averaging the second to last layer of the output and returning the first 5 features of the embedding
    
    return sentence_embedding[:5]


# Adding each sentence embedding to their respective list
for item in combined_list:
    nc_orignal_sentence_vector.append(sentence_embeddings(item[0]))
    syn_orignal_sentence_vector.append(sentence_embeddings(item[1]))
    


combined_sentence_embeddings = zip(nc_orignal_sentence_vector,syn_orignal_sentence_vector)

# Working out the cosine sim between the NC and its synonm
for item in combined_sentence_embeddings:
    vector_simi = 1 - cosine(item[0],item[1])
    all_vector_simi.append(vector_simi)
    

vector_embeddings_with_NC = zip(nc_pd,all_vector_simi)



with open('nc_syn_sentence_change.csv','wt') as out_file:
    headersList =['NC','Vector Sim value (with syn)']
    tsv_writer = csv.writer(out_file)
    tsv_writer.writerow(headersList)
    for NC,vector_val in vector_embeddings_with_NC:
        tsv_writer.writerow([NC, vector_val])

