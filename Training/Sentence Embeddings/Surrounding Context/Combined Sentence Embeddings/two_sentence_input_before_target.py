# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 04:50:13 2022

@author: James
"""

# =============================================================================
# This code generates an average for all features in the sentence embeddings when:
# The sentences before the idiom and the idiom are inputted into BERT 
# =============================================================================


import csv
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Stores the key as an incrementing value by one and the value as the sentence embedding
sentence_before_and_target_dict = {}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sem_eval_dataset = pd.read_csv('sem_eval_train_one_shot.csv')

# Combines the sentences before and the target sentences to create a 2D array of them both
target_sentences = sem_eval_dataset.query('Label==0')['Target']
sentences_before = sem_eval_dataset.query('Label==0')['Previous']
sentence_before_and_target = zip(sentences_before,target_sentences)



# Code inspired from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
def sentence_embedding_two_inputs(text,text2):
    # Using the special syntax (SEP) to create a more contextual embeddings by adding two sentences
    marked_text = "[CLS] " + text + " [SEP] " + text2
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


# Generates the sentence embeddings for the whole sentence and outputs it to the dictionary
i = 0
for before,target in sentence_before_and_target:
    i = i + 1
    sentence_vector = sentence_embedding_two_inputs(before,target)
    sentence_before_and_target_dict[i] = sentence_vector




with open('two_sentence_input_embeddings_before_target.csv','wt', encoding='UTF8') as out_file:
    headersList =['Sentence_num','Sentence Vector']
    tsv_writer = csv.writer(out_file)
    tsv_writer.writerow(headersList)
    for sent_id,vector in sentence_before_and_target_dict.items():
        tsv_writer.writerow([sent_id, vector])

