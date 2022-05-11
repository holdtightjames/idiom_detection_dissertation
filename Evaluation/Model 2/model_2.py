# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 03:30:08 2022

@author: James
"""

# =============================================================================
# This is model 2 that implements:
# word embeddings features
# sentiment features 
# term frequency feature
# word type majority feature
# Sentence Embeddings feature
# =============================================================================

from nltk.tokenize import  word_tokenize
import json
from itertools import islice
from nltk.corpus import stopwords
import csv, nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import math
from scipy.spatial.distance import cosine
import torch
from transformers import BertTokenizer, BertModel


phrase_idiom_val_dict = {}
#Dictionary that holds the sentence ID's and the idiomatic value
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Tokenizer that has a large dictionary of tokenised words
gold_standard_magpie_testing = {}
#Dictionary that holds the true values for the dataset
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
# Used for the stop list when doing term frequency
stop_word_dictionary = stopwords.words('english')


with open('MAGPIE_filtered_split_typebased.jsonl', 'r', encoding='utf-8') as json_file:
    Magpie_corpus = list(json_file)

for line in Magpie_corpus:
    Magpie_corpus_loaded = json.loads(line)
    if Magpie_corpus_loaded['split'] == "test":
        gold_standard_magpie_testing [Magpie_corpus_loaded['id']] = (Magpie_corpus_loaded['context'])
        # Used to generate the dictionary containing the context and sentence ID of the test data in MAGPIE
    

# Function to check the word type majority of a sentence
def word_type_detector(sentence):
    temp_dict = {}
    adjective = 0
    noun = 0
    adverb = 0
    verb = 0
    temp_dict.clear()
    text_to_assess = word_tokenize(sentence)
    val = nltk.pos_tag(text_to_assess)
    for item in val:
        if item[1] == "JJ" or item[1] == "JJR" or item[1] == "JJS":
            adjective = adjective + 1
        elif item[1] == "NN" or item[1] == "NNP" or item[1] == "NNS":
            noun = noun + 1
        elif item[1] == "RB" or item[1] == "RBR" or item[1] == "RBS":
            adverb = adverb + 1
        elif item[1] == item[1] == "VB" or item[1] == "VBD" or item[1] == "VBG" or item[1] == "VBN" or item[1] == "VBP" or item[1] == "VBZ":
            verb = verb + 1
        # Checks the type of each word in a sentence and adds it to a cumulative dictionary
    temp_dict ['adjective'] = adjective
    temp_dict ['noun'] = noun
    temp_dict ['adverb'] = adverb
    temp_dict ['verb'] = adverb
    sorted_dictionary_tfidf = sorted(temp_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_dictionary_tfidf[0][0]
    # Returns the string of the majority used word type


# Takes a sentence input and returns idiom value of 40 if it is negative overall.
def sent_score(line):
    sia = SentimentIntensityAnalyzer()
    if (sia.polarity_scores(line)["compound"] > 0):
        idiom_value = 10
    else:
        idiom_value = 40
    return idiom_value


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
            gold_standard_magpie.append(Magpie_corpus_loaded['idiom'])
            # Gathers the term frequency of the words in the training split of the MAGPIE dictionary
    for line in gold_standard_magpie:
        if line not in overlap_stopping_list:
            # Checks the idiom has not been used before
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


# Checks each context for sentiment and inclusion of words in the term frequency dicionary
def tf_sent_process():
    term_tf_dict = idiom_tf()
    for sent_id,context in islice(gold_standard_magpie_testing.items(), 0,100):
        flattened_list = ' '.join(context)
        amount_word_features = 0
        cumalative_sentiment = 0
        word_type_scalar = 0
        for word in flattened_list.split():
            if word in term_tf_dict:
                amount_word_features = amount_word_features + (term_tf_dict.get(word) * 10)
                # Scales it up by * 10 of the words count in the dictionary
        for sentence in context:
            idiom_value = sent_score(sentence)
            cumalative_sentiment = idiom_value + cumalative_sentiment
        check_word_type = word_type_detector (flattened_list)
        if check_word_type == "noun":
            word_type_scalar = 70
            # Applies an added value to the models overall idiomatic value if the majority word type is a noun
        if sent_id in phrase_idiom_val_dict:
            phrase_idiom_val_dict[sent_id] = phrase_idiom_val_dict.get(sent_id) + cumalative_sentiment + amount_word_features + word_type_scalar
            # Combines all the weightings of the sentiment and tf and then adds it to the dictionary of sentence ID and value
        else:
            phrase_idiom_val_dict[sent_id] = cumalative_sentiment + amount_word_features + word_type_scalar
    return phrase_idiom_val_dict


# Function is used to generate word embeddings
# Based of https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/ code
def noun_compound_vector(text):
    token_vecs_sum_gen = []
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
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum_gen.append(sum_vec)
    return token_vecs_sum_gen


# Check to see if any words have similiar cosine similiarity of NC embeddings
def word_vector_calc():
    for sent_id,context in islice(gold_standard_magpie_testing.items(), 0,100):
        flattened_list = ' '.join(context)
        literally_factor = 0
        if "literally" in flattened_list:
            # If the word 'literally' is in the context then it is less likely to be idiomatic
            literally_factor = -300
        NC_diff_list = []
        current_state = "NULL"
        token_vecs_sum = noun_compound_vector(flattened_list)
        increased_idiom_value = 0
        location_of_pot_idiom = "NULL"
        for i in range (0, len(flattened_list.split())):
            NC_diff = 1 - cosine(token_vecs_sum[i], token_vecs_sum[i+1])
            # Checks the current words vector against the next words vector
            NC_diff_list.append(NC_diff)
            for numbers in NC_diff_list:
                if current_state != "precise":
                    # The following code checks if it is within the thresholds generated from training data
                    if numbers > 0.55 or numbers < 0.45:
                        if current_state == "closer":
                            current_state = "closer"
                        else:
                            current_state = "not close"
                    elif 0.51 >= numbers >= 0.47:
                        if 49 <= numbers <= 50:
                            current_state = "precise"
                            if i < math.floor((len(flattened_list.split()))/4):
                                location_of_pot_idiom = "start"
                            elif i > math.floor((len(flattened_list.split())*3)/4):
                                # Checks the location the potential NC was found in
                                location_of_pot_idiom = "end"
                            if (math.floor(len(flattened_list.split()))/4) < i < math.floor((len(flattened_list.split()) * 3)/4):
                                location_of_pot_idiom = "middle"
                        else:
                          current_state = "closer"
                          if i < math.floor((len(flattened_list.split()))/4):
                            location_of_pot_idiom = "start"
                          elif i > math.floor((len(flattened_list.split())*3)/4):
                            location_of_pot_idiom = "end"
                          if (math.floor(len(flattened_list.split()))/4) < i < math.floor((len(flattened_list.split()) * 3)/4):
                            location_of_pot_idiom = "middle"
                    else:
                      current_state = "closer"
                      if i < (len(flattened_list.split()))/4:
                          location_of_pot_idiom = "start"
                      elif i > (len(flattened_list.split()) * 3)/4:
                          location_of_pot_idiom = "end"
                      if ((len(flattened_list.split()))/4) < i < ((len(flattened_list.split()) * 3)/4):
                          location_of_pot_idiom = "middle"
        if current_state == "closer":
            idiom_value = 10
        elif current_state == "precise":
            # largest increase to the idiomatic value here as it is closest to the threshold data generated from training
            idiom_value = 20
        elif current_state == "not close":
            idiom_value = 3
        if location_of_pot_idiom == "start":
            idiom_value = idiom_value + 10
        elif location_of_pot_idiom == "middle":
            idiom_value = idiom_value + 5
        elif location_of_pot_idiom == "end":
            idiom_value = idiom_value + 25
        if sent_id in phrase_idiom_val_dict:
            phrase_idiom_val_dict [sent_id] = idiom_value + phrase_idiom_val_dict.get(sent_id) + increased_idiom_value + literally_factor
        else:
            phrase_idiom_val_dict [sent_id] = idiom_value + increased_idiom_value + literally_factor


# Generates sentence embeddings for the vectors\
# Based on code of https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
def noun_compound_vector_sentence(text,text2,two_input):
    if two_input == True:
        marked_text = "[CLS] " + text + " [SEP] " + text2
    else:
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
    return sentence_embedding[:5]

# Used to find the average value between all the features of the vector
def tensor_flow_convert(vector_value):
    cumalative_score = 0
    begining_string = str(vector_value).find('[')
    end_of_string = str(vector_value).find(']')
    new_word = str(vector_value)[begining_string + 1:end_of_string]
    all_string_item = new_word.split(',')
    for number_in_vector in all_string_item:
        cumalative_score = float(number_in_vector) + cumalative_score
    average_vector = cumalative_score/5
    return average_vector
    

# Finds the average value of all the feature vectors and check against thresholds of training
def average_sentence_vector():
    for sent_id,context in islice(gold_standard_magpie_testing.items(), 0,100):
        idiom_value = 0
        average_sentence_idiom_chance = 0
        for item in context:
            average_vector_val = 0
            sentence_vector = noun_compound_vector_sentence(item,"NULL", False)
            average_vector_val = tensor_flow_convert(str(sentence_vector))
            if average_vector_val > 0.06 or average_vector_val < 0.01:
                average_sentence_idiom_chance = average_sentence_idiom_chance + 1
            else:
                if 0.030 <= average_vector_val <= 0.038:
                    if 0.032 <= average_vector_val <= 0.036:
                        average_sentence_idiom_chance = average_sentence_idiom_chance + 10
                    else:
                        average_sentence_idiom_chance = average_sentence_idiom_chance + 7
                else:
                    average_sentence_idiom_chance = average_sentence_idiom_chance + 2
        if average_sentence_idiom_chance >= 10:
            idiom_value = 5
        if average_sentence_idiom_chance >= 15:
            idiom_value = 10
        if average_sentence_idiom_chance >= 20:
            idiom_value = 20
        if sent_id in phrase_idiom_val_dict:
            phrase_idiom_val_dict [sent_id] = idiom_value + phrase_idiom_val_dict.get(sent_id)
        else:
            phrase_idiom_val_dict [sent_id] = idiom_value
        

# Checks between the sentence vectors and the cosine similiarity of them.
# The vectors used will be from the sentence before a potential idiom and the idiom and the sentence after the potential idiom.
def pipelined_sentence():
    sentence_one = "NULL"
    sentence_two = "NULL"
    sentence_four = "NULL"
    sentence_five = "NULL"
    for sent_id,context in islice(gold_standard_magpie_testing.items(), 0,100):
        current_state = "NULL"
        idiom_value = 0
        for item in context:
            if context.index(item) < (len(context) - 1):
                sentence_one = context[context.index(item)]
                sentence_two = context[context.index(item) + 1]
            sentence_vector_before = noun_compound_vector_sentence(sentence_one,"NULL", False)
            sentence_vector_after = noun_compound_vector_sentence(sentence_two,"NULL", False)
            cosine_in_vector_after = 1 - cosine(sentence_vector_before, sentence_vector_after)
            if cosine_in_vector_after > 0.50 or cosine_in_vector_after < 0.30:
                if current_state != "precise":
                    if current_state == "maybe_idiom":
                        current_state = "maybe_idiom"
                    else:
                        current_state = "probably_not_idiom"
            elif 0.47 >= cosine_in_vector_after >= 0.37:
                if 0.37 <= cosine_in_vector_after <= 0.43:
                    current_state = "precise"
                else:
                    current_state = "maybe_idiom"
            if (context.index(item) > 1):
                sentence_four = context[context.index(item) - 1]
                sentence_five = context[context.index(item)]
            sentence_vector_before_2 = noun_compound_vector_sentence(sentence_four,"NULL", False)
            sentence_vector_after_2 = noun_compound_vector_sentence(sentence_five,"NULL", False)
            cosine_in_vector_before_2 = 1 - cosine(sentence_vector_before_2, sentence_vector_after_2)
            if cosine_in_vector_before_2 > 0.55 or cosine_in_vector_before_2 < 0.30:
                if current_state != "precise":
                    if current_state == "maybe_idiom":
                        current_state = "maybe_idiom"
                    else:
                        current_state = "probably_not_idiom"
            elif 0.40 >= cosine_in_vector_before_2 >= 0.33:
                if 0.34 <= cosine_in_vector_before_2 <= 0.39:
                    current_state = "precise"
                else:
                    current_state = "maybe_idiom"
        if current_state == "maybe_idiom":
            idiom_value = 10
        elif current_state == "precise":
            idiom_value = 20
        elif current_state == "probably_not_idiom":
            idiom_value = 3
        if sent_id in phrase_idiom_val_dict:
            phrase_idiom_val_dict [sent_id] = idiom_value + phrase_idiom_val_dict.get(sent_id)
        else:
            phrase_idiom_val_dict [sent_id] = idiom_value


# More contextual embeddings by finding when contexts are applied together. 
# Compares to the thresholds generated in training
def Bert_two_sentence_input():
    sentence_one = "NULL"
    sentence_two = "NULL"
    sentence_one_after = "NULL"
    sentence_two_after = "NULL"
    for sent_id,context in islice(gold_standard_magpie_testing.items(), 0,100):
        average_sentence_idiom_chance = 0
        idiom_value = 0
        for item in context:
            if context.index(item) > 1:
                sentence_one = context[context.index(item) - 1]
                sentence_two = context[context.index(item)]
                sentence_vector = noun_compound_vector_sentence(sentence_one,sentence_two, True)
                average_sentence_vector = tensor_flow_convert(sentence_vector)
                if average_sentence_vector > 0.050 or average_sentence_vector < 0.020:
                    average_sentence_idiom_chance = average_sentence_idiom_chance + 1
                else:
                    if 0.032 <= average_sentence_vector <= 0.04:
                        if 0.033 <= average_sentence_vector <= 0.039:
                            average_sentence_idiom_chance = average_sentence_idiom_chance + 15
                        else:
                            average_sentence_idiom_chance = average_sentence_idiom_chance + 7
                    else:
                        average_sentence_idiom_chance = average_sentence_idiom_chance + 5
            if context.index(item) < (len(context) - 1):
                sentence_one_after = context[context.index(item)]
                sentence_two_after = context[context.index(item) + 1]
                sentence_vector_after = noun_compound_vector_sentence(sentence_one_after,sentence_two_after, True)
                average_sentence_vector_after = tensor_flow_convert(sentence_vector_after)
                if average_sentence_vector_after > 0.04 or average_sentence_vector_after < 0.01:
                    average_sentence_idiom_chance = average_sentence_idiom_chance + 1
                else:
                    if 0.019 <= average_sentence_vector_after <= 0.027:
                        if 0.020 <= average_sentence_vector_after <= 0.026:
                            average_sentence_idiom_chance = average_sentence_idiom_chance + 15
                        else:
                            average_sentence_idiom_chance = average_sentence_idiom_chance + 7
                    else:
                        average_sentence_idiom_chance = average_sentence_idiom_chance + 5
        if average_sentence_idiom_chance >= 15:
            idiom_value = 5
        if average_sentence_idiom_chance >= 20:
            idiom_value = 10
        if average_sentence_idiom_chance >= 25:
            idiom_value = 15
        if sent_id in phrase_idiom_val_dict:
            phrase_idiom_val_dict [sent_id] = idiom_value + phrase_idiom_val_dict.get(sent_id)
        else:
            phrase_idiom_val_dict [sent_id] = idiom_value

    
tf_sent_process()
word_vector_calc()
average_sentence_vector()
pipelined_sentence()
Bert_two_sentence_input()


# Holds the sentence ID and the prediction
ouput_dict = {}
# Checks the cumulative value to values generated in training to see if the value should be an idiom or not.
for sent_id,idiom_val in phrase_idiom_val_dict.items():
    if 1470 >= idiom_val >= 450:
        output_string = "i"
    else:
        output_string = "l"
    ouput_dict [sent_id] = output_string
  
        
with open('magpie_output.csv','wt') as out_file:
    headersList =['idiom','Prediction']
    tsv_writer = csv.writer(out_file)
    tsv_writer.writerow(headersList)
    for sent_id,idiom_pred in ouput_dict.items():
        tsv_writer.writerow([sent_id, idiom_pred])


    