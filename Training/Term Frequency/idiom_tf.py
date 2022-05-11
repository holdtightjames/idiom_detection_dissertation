# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 02:42:53 2022

@author: James
"""

# =============================================================================
# Code to check the common terms within idioms in the MAGPIE training data
# =============================================================================


import json
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
# Using the NLTK stopwords to remove common words within idioms that wont add much change

stop_word_dictionary = stopwords.words('english')
# Creates a variable for the stop words

delete_list =[]
#Stores all the terms in the stop list from NLTK

gold_standard_magpie = []
# All the idioms in the training split

term_tf_dict = {}
# Dictionary storing the term as the key and the frequency as the value

overlap_stopping_list = []
# Variable to store unqiue idiom instances (compare to this list to remove items)


with open('MAGPIE_filtered_split_typebased.jsonl', 'r', encoding='utf-8') as json_file:
    Magpie_corpus = list(json_file)

for line in Magpie_corpus:
    Magpie_corpus_loaded = json.loads(line)
    if Magpie_corpus_loaded['split'] == "training":
        # Only uses idioms from the training set to ensure integrity is kept
        gold_standard_magpie.append(Magpie_corpus_loaded['idiom'])
for line in gold_standard_magpie:
    if line not in overlap_stopping_list:
        # Checks there is not multiple instances of the same idiom
        for term in (line.split(' ')):
            if term in term_tf_dict:
                # Increments the term amount
                term_tf_dict[term] = term_tf_dict.get(term) + 1
            else:
                term_tf_dict[term] = 1
        overlap_stopping_list.append(line)
for key in term_tf_dict.keys():
    if key in stop_word_dictionary:
        delete_list.append(key)
for words in delete_list:
    term_tf_dict.pop(words)
    # Removes words that are in the stop list
print(term_tf_dict)

