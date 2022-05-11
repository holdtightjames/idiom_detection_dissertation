# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 04:50:13 2022

@author: James
"""
# =============================================================================
# Code to work out the average feature value of neutral sentences such as:
# This is a noise complaint
# =============================================================================


import csv
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel


# List to hold all the sentence embeddings of the neutral sentences
sentence_vector = []


# Pre-trained model that tokenises the words based on a large vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Retrieving neutral sentences for idioms
neutral_dataset = pd.read_csv('P1_sents.tsv')
neutral_sentence = neutral_dataset["neutral sentence"]


# Code inspired from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
def sentence_embeddings_generator(text):

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


# Creates a list for all the sentence embeddings of the neutral sentences
for item in neutral_sentence:
    sentence_vector.append(sentence_embeddings_generator(item))
    

# Combines orignal sentence with their embeddings
combined_sentence_embeddings = zip(neutral_sentence,sentence_vector)




# Outputs to a tsv file to be examined later for the average feature values
with open('neutral_sentence_embeddings.tsv','wt') as out_file:
    headersList =['Sentence','Sentence Vector']
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(headersList)
    for NC,vector_val in combined_sentence_embeddings:
        tsv_writer.writerow([NC, vector_val])
