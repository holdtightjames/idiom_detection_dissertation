# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 04:50:13 2022

@author: James
"""

# =============================================================================
# Code to see the sentence embeddings of a contextual sentence such as:
# "Program leaders said the scholarship defines public service broadly and imagines a variety of pathways toward civic engagement."
# =============================================================================

import csv
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel


# List to hold all the sentence embeddings of the sentences
sentence_vector = [] 

# Pre-trained model that tokenises the words based on a large vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Importing just the idiom sentences from the dataset
sem_eval_dataset = pd.read_csv('sem_eval_train_one_shot.csv')
context_sentences = sem_eval_dataset.query('Label==0')['Target']


# Code inspired from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
def sentence_embeddings(text):
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
    sentence_embedding = torch.mean(token_vecs, dim=0)
    # Averaging the second to last layer of the output and returning the first 5 features of the embedding
    return sentence_embedding[:5]


# Iterates through the dataset passing all the sentences into the sentence_embeddings function
for item in context_sentences:
    sentence_vector.append(sentence_embeddings(item))
    

# Adds all the sentence embeddings to the original sentence in a new list
combined_sentence_embeddings = zip(context_sentences,sentence_vector)



# Writes all the sentence embeddings and the orignal sentence to a tsv file
with open('context_sentence_embeddings.tsv','wt', encoding='UTF8') as out_file:
    headersList =['Sentence','Sentence Vector']
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(headersList)
    for NC,vector_val in combined_sentence_embeddings:
        tsv_writer.writerow([NC, vector_val])