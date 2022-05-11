# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:56:30 2022

@author: James
"""

# =============================================================================
# Identifying the majority word type using lexcomp dataset and averaged_perceptron_tagger
# =============================================================================

import nltk
import json
from nltk.tokenize import  word_tokenize

nltk.download('averaged_perceptron_tagger')
#Used to tag the words in the sentence with a word type
lexcomp_dataset = []
# Stores the idiomatic sentences
lexcomp_loaded = {}
# Stores the whole lexcomp dataset
temp_dict = {}
# Temporarily stores the word type and the freq in the sentence
majority_word_type_dict = {}
# Stores the majority word type as the key and the value is the freq

with open('train.jsonl', 'r', encoding='utf-8') as json_file:
    lexcomp = list(json_file)
for line in lexcomp:
    lexcomp_loaded = json.loads(line)
    if lexcomp_loaded['label'] == "NON-LITERAL":
        lexcomp_dataset.append(lexcomp_loaded['sentence'])


for item in lexcomp_dataset:
    # Iterates through all the sentences
    adjective = 0
    noun = 0
    adverb = 0
    verb = 0
    temp_dict.clear()
    text = word_tokenize(item)
    val = nltk.pos_tag(text)
    for item in val:
        # Checks each word to see if it has been labelled with a specific word type using NLTK
        if item[1] == "JJ" or item[1] == "JJR" or item[1] == "JJS":
            adjective = adjective + 1
        elif item[1] == "NN" or item[1] == "NNP" or item[1] == "NNS":
            noun = noun + 1
        elif item[1] == "RB" or item[1] == "RBR" or item[1] == "RBS":
            adverb = adverb + 1
        elif item[1] == item[1] == "VB" or item[1] == "VBD" or item[1] == "VBG" or item[1] == "VBN" or item[1] == "VBP" or item[1] == "VBZ":
            verb = verb + 1
    temp_dict ['adjective'] = adjective
    temp_dict ['noun'] = noun
    temp_dict ['adverb'] = adverb
    temp_dict ['verb'] = verb
    sorted_dictionary_tfidf = sorted(temp_dict.items(), key=lambda x: x[1], reverse=True)
    # Takes the biggest majority type from the sentence being iterated
    if sorted_dictionary_tfidf[0][0] in majority_word_type_dict:
        majority_word_type_dict [sorted_dictionary_tfidf[0][0]] = majority_word_type_dict.get(sorted_dictionary_tfidf[0][0]) + 1
        # Adds the majority word type to another dictionary that is counting the majority type for all sentences
    else:
        majority_word_type_dict [sorted_dictionary_tfidf[0][0]] = 1
        
        
print(majority_word_type_dict)
    
        
    
    
        