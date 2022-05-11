# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 04:50:13 2022

@author: James
"""

# =============================================================================
# This code will work out the cosine similiarity between sentence embeddings before the idiom and the sentence containing
# the idiom
# =============================================================================
import csv
import pandas as pd
from scipy.spatial.distance import cosine
import torch
from transformers import BertTokenizer, BertModel



sentence_before_and_target = []
# This list will store the sentences before and the target

before_target_vector_dict = {}
# This dictionary will store the cosine similiarities for each record



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



sem_eval_dataset = pd.read_csv('sem_eval_train_one_shot.csv')

target_sentences = sem_eval_dataset.query('Label==0')['Target']
sentences_before = sem_eval_dataset.query('Label==0')['Previous']

sentence_before_and_target = zip(sentences_before,target_sentences)



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

i = 0

for before,target in sentence_before_and_target:
    i = i + 1
    before_vectors = sentence_embeddings(before)
    target_vectors = sentence_embeddings(target)
    # Calculates the cosine similiarity between the vectors before the idiom and the idiom sentence
    cosine_sentence_sim = 1 - cosine(before_vectors, target_vectors) 
    before_target_vector_dict[i] = cosine_sentence_sim




with open('before_target_cosine.csv','wt') as out_file:
    headersList =['Sentence_num','Cosine Sim']
    tsv_writer = csv.writer(out_file)
    tsv_writer.writerow(headersList)
    for sent_id,cosine_sentence_sim in before_target_vector_dict.items():
        tsv_writer.writerow([sent_id, cosine_sentence_sim])