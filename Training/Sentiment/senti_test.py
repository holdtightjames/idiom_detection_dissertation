# =============================================================================
# # Code to test the overall idiom sentiment of Static_idioms.txt
# =============================================================================

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

gold_standard_magpie = []
positive_negative_dictionary = {}
idiom_list = []



file1 = open('Static_idioms.txt', 'r')
for line in file1:
    idiom_list.append(line.strip())
    
    
# Code inspired from the sentiment analysis library
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
for item in idiom_list:
    if (sia.polarity_scores(item)["compound"] > 0):
        #Checks if overall sentiment is positive
        if 'pos' in positive_negative_dictionary:
            positive_negative_dictionary['pos'] = positive_negative_dictionary.get('pos') + 1
        else:
            positive_negative_dictionary['pos'] = 1
    else:
        #Checks if overall sentiment is negative
        if 'neg' in positive_negative_dictionary:
            positive_negative_dictionary['neg'] = positive_negative_dictionary.get('neg') + 1
        else:
            positive_negative_dictionary['neg'] = 1
        
    
        
print(positive_negative_dictionary)
# {'neg': 310, 'pos': 49}


