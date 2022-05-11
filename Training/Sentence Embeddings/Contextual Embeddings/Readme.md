This folder contains the feature vector when a contextual sentence containing an idiom is evaluated.

context_sentence_embeddings.tsv holds the sentence embeddings of each sentence and the sentence. This is calculated using context_sentence_embeddings.py

feature_embeddings_average_calc.py uses the context_sentence_embeddings.tsv file to find an average for the feature vectors.

The output of this is stored in the context_features_mean.csv file.

The dataset used here is sem_eval as stated in the paper