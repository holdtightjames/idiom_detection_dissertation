# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 03:30:08 2022

@author: James
"""
# =============================================================================
# This is model 1 that applies the word embeddings features, sentiment features and term frequency feature
# =============================================================================

import pandas as pd
import json
from nltk.corpus import stopwords
import csv, nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import math
from scipy.spatial.distance import cosine
import torch
from transformers import BertTokenizer, BertModel


phrase_idiom_val_dict = {}



nltk.download('stopwords')
stop_word_dictionary = stopwords.words('english')
wiktionary_dataset = pd.read_csv('wiktionary_dataset.csv')
idioms = wiktionary_dataset['IDIOM']


# Using the bert-base-uncased that tokenises the words based on the large vocabulary
# If there is no word, it will split it up
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')




# Finds the sentiment of the sentence and returns a higher value if the sentence is negative
def sent_score(line):
    sia = SentimentIntensityAnalyzer()
    if (sia.polarity_scores(line)["compound"] > 0):
        idiom_value = 0
    else:
        idiom_value = 10
    return idiom_value


# Generates the tf dicionary using the MAGPIE training set.
# Uses a stop list to elimnate frequent words
def idiom_tf():
    delete_list =[]
    gold_standard_magpie = []
    masked_idiom_list = []
    predicted_sentences = []
    term_tf_dict = {}
    overlap_stopping_list = []
    with open('MAGPIE_filtered_split_typebased.jsonl', 'r', encoding='utf-8') as json_file:
        Magpie_corpus = list(json_file)
    for line in Magpie_corpus:
        Magpie_corpus_loaded = json.loads(line)
        if Magpie_corpus_loaded['split'] == "training":
            # Keeps integrity by using seperate dataset split to train the model compared to the evaluation
            gold_standard_magpie.append(Magpie_corpus_loaded['idiom'])
    for line in gold_standard_magpie:
        if line not in overlap_stopping_list:
            # Checks that each idiom counted is unique and no duplicates
            for term in (line.split(' ')):
                if term in term_tf_dict:
                    term_tf_dict[term] = term_tf_dict.get(term) + 1
                else:
                    term_tf_dict[term] = 1
            overlap_stopping_list.append(line)
    for key in term_tf_dict.keys():
        if key in stop_word_dictionary:
            delete_list.append(key)
    for words in delete_list:
        term_tf_dict.pop(words)
    return term_tf_dict


# Checks the sentiment and tf of each input
def tf_sent_process():
    term_tf_dict = idiom_tf()
    for item in idioms:
        amount_word_features = 0
        idiom_value = sent_score(item)
        if item in phrase_idiom_val_dict:
            phrase_idiom_val_dict[item] = phrase_idiom_val_dict.get(item) + idiom_value
        else:
            phrase_idiom_val_dict[item] = idiom_value
        for words in item.split():
            if words in term_tf_dict:
                amount_word_features = amount_word_features + (term_tf_dict.get(words) * 30)
                # Applied a scalar to increase the importance of tf in the model
        if item in phrase_idiom_val_dict:
            phrase_idiom_val_dict [item] = phrase_idiom_val_dict.get(item) + amount_word_features
        else:
            phrase_idiom_val_dict [item] = amount_word_features
    return phrase_idiom_val_dict


                          

# Generates word embeddings.
# Based on work from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/ code
def word_embeddings_generator(text):
    
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
    # Changes the order of the embeddings where the last column is the features
    token_vecs = hidden_states[-2][0]
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        # Sums the last 4 layers of BERT to generate word embeddings
        token_vecs_sum_gen.append(sum_vec)
    return token_vecs_sum_gen



# Checks the word embeddings against the ones generated in training
# Increases the idiomatic value accordingly
for item in idioms[:100]:
    increased_idiom_value = 0
    NC_diff_list = []
    current_state = "NULL"
    token_vecs_sum = word_embeddings_generator(item)
    location_of_pot_idiom = "NULL"
    for i in range (0, len(item.split())):
        NC_diff = 1 - cosine(token_vecs_sum[i], token_vecs_sum[i+1])
        NC_diff_list.append(NC_diff)
        for numbers in NC_diff_list:
            if current_state != "precise":
                # Checks wether the cosine sim between words are similiar to the ones generated in training
                if numbers > 0.55 or numbers < 0.45:
                    if current_state == "closer":
                        current_state = "closer"
                    else:
                        current_state = "not close"
                elif 0.51 >= numbers >= 0.47:
                    if  0.49 <= numbers <= 0.50:
                        current_state = "precise"
                        if i < math.floor((len(item.split()))/4):
                            # Checks the location of the potential idioms
                            location_of_pot_idiom = "start"
                        elif i > math.floor((len(item.split())*3)/4):
                            location_of_pot_idiom = "end"
                        if (math.floor(len(item.split()))/4) < i < math.floor((len(item.split()) * 3)/4):
                            location_of_pot_idiom = "middle"
                    else:
                        current_state = "closer"
                        if i < (len(item.split()))/4:
                            location_of_pot_idiom = "start"
                        elif i > (len(item.split()) * 3)/4:
                            location_of_pot_idiom = "end"
                        if ((len(item.split()))/4) < i < ((len(item.split()) * 3)/4):
                            location_of_pot_idiom = "middle"
    if current_state == "closer":
        idiom_value = 10
    elif current_state == "precise":
        idiom_value = 20
    elif current_state == "not close":
        idiom_value = 3
    if location_of_pot_idiom == "start":
        idiom_value = idiom_value + 10
    elif location_of_pot_idiom == "middle":
        idiom_value = idiom_value + 15
    elif location_of_pot_idiom == "end":
        idiom_value = idiom_value + 5
    if item in phrase_idiom_val_dict:
        phrase_idiom_val_dict [item] = idiom_value + phrase_idiom_val_dict.get(item) + increased_idiom_value
    else:
        phrase_idiom_val_dict [item] = idiom_value + increased_idiom_value
        
tf_sent_process()

ouput_dict = {}
# Checks the idiom value of each sentence against the threshold generated in training
for sentence,idiom_val in phrase_idiom_val_dict.items():
    if idiom_val >= 95:
        output_string = "idiom"
    else:
        output_string = "literal"
    ouput_dict [sentence] = output_string
    

    
# Outputs the idiom and the prediction. where an idiom is labelled 'idiom' and literal is 'literal'
# This will be checked by the wiktionary calculator 
with open('wiktionary_output.csv','wt') as out_file:
    headersList =['idiom','Prediction']
    tsv_writer = csv.writer(out_file)
    tsv_writer.writerow(headersList)
    for sentence,idiom_pred in ouput_dict.items():
        tsv_writer.writerow([sentence, idiom_pred])
    
    

    
    
