import numpy as np
import pandas as pd
import pytrec_eval
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import brown
import pickle

nltk.data.path.append('./NLTK_Data/')


# This function takes in the error corpus sentence and returns the top 10 predicted next word using:
#   1-Gram language model
#   2-Gram language model
#   3-Gram language model
#   5-Gram language model
#   10-Gram language model
def pred_word(given_corpus):
    _1gram_prob = pickle.load(open('1-Gram-Model.pkl', 'rb'))
    _2gram_prob = pickle.load(open('2-Gram-Model.pkl', 'rb'))
    _3gram_prob = pickle.load(open('3-Gram-Model.pkl', 'rb'))
    _5gram_prob = pickle.load(open('5-Gram-Model.pkl', 'rb'))
    _10gram_prob = pickle.load(open('10-Gram-Model.pkl', 'rb'))

    ngram1 = tuple(given_corpus[-1:])
    ngram2 = tuple(given_corpus[-2:])
    ngram3 = tuple(given_corpus[-3:])
    ngram5 = tuple(given_corpus[-5:])
    ngram10 = tuple(given_corpus[-10:])

    ngram1_len = len(ngram1)
    ngram2_len = len(ngram2)
    ngram3_len = len(ngram3)
    ngram5_len = len(ngram5)
    ngram10_len = len(ngram10)

    pred_1gram = []
    pred_2gram = []
    pred_3gram = []
    pred_5gram = []
    pred_10gram = []

    count = 0

    for each in _1gram_prob:
        if each[0][:-1] == ngram1:
            count += 1
            pred_1gram.append(each[0][-1])
            if count == 10:
                break

    if count < 10:
        while count != 10:
            pred_1gram.append("NF")
            count += 1

    count = 0
    for each in _2gram_prob:
        if each[0][0:ngram2_len] == ngram2:
            count += 1
            pred_2gram.append(each[0][ngram2_len])
            if count == 10:
                break

    if count < 10:
        while count != 10:
            pred_2gram.append("NF")
            count += 1

    count = 0
    for each in _3gram_prob:
        if each[0][0:ngram3_len] == ngram3:
            count += 1
            pred_3gram.append(each[0][ngram3_len])
            if count == 10:
                break

    if count < 10:
        while count != 10:
            pred_3gram.append("NF")
            count += 1

    count = 0
    for each in _5gram_prob:
        if each[0][0:ngram5_len] == ngram5:
            count += 1
            pred_5gram.append(each[0][ngram5_len])
            if count == 10:
                break

    if count < 10:
        while count != 10:
            pred_5gram.append("NF")
            count += 1

    count = 0
    for each in _10gram_prob:
        if each[0][0:ngram10_len] == ngram10:
            count += 1
            pred_10gram.append(each[0][ngram10_len])
            if count == 10:
                break

    if count < 10:
        while count != 10:
            pred_10gram.append("NF")
            count += 1

    return pred_1gram, pred_2gram, pred_3gram, pred_5gram, pred_10gram


corpus_df = pd.read_excel("C://Users//lavee//Documents//UWindsor//Courses//Winter'22//NLP//Assignment 2//Corpus.xlsx",
                          index_col=None)

corpus_df = corpus_df.head(1)
correct_words_list = corpus_df['Solution'].tolist()

corpus_list = corpus_df[corpus_df.columns[1:6]]
corpus_list = corpus_list.to_records(index=False)
corpus_list = list(corpus_list)

corpus_list_cleaned = []

for words in corpus_list:
    newlist = [x for x in words if pd.isnull(x) == False]
    corpus_list_cleaned.append(newlist)

s1_list_1gram = []
s1_list_2gram = []
s1_list_3gram = []
s1_list_5gram = []
s1_list_10gram = []

s5_list_1gram = []
s5_list_2gram = []
s5_list_3gram = []
s5_list_5gram = []
s5_list_10gram = []

s10_list_1gram = []
s10_list_2gram = []
s10_list_3gram = []
s10_list_5gram = []
s10_list_10gram = []
count = 0

# Loop through each sentence in the error corpus
for words, correct_word in zip(corpus_list_cleaned, correct_words_list):
    print(count)
    count = count + 1
    # Get the next word predictions for the given sentence
    _1gram_prediction, _2gram_prediction, _3gram_prediction, _5gram_prediction, _10gram_prediction = pred_word(words)
    print(_1gram_prediction)

    # Check if word appears in the first prediction to calculate s@1
    if correct_word == _1gram_prediction[0]:
        s_1 = 1
    else:
        s_1 = 0
    s1_list_1gram.append(s_1)
    if correct_word == _2gram_prediction[0]:
        s_1 = 1
    else:
        s_1 = 0
    s1_list_2gram.append(s_1)
    if correct_word == _3gram_prediction[0]:
        s_1 = 1
    else:
        s_1 = 0
    s1_list_3gram.append(s_1)
    if correct_word == _5gram_prediction[0]:
        s_1 = 1
    else:
        s_1 = 0
    s1_list_5gram.append(s_1)
    if correct_word == _10gram_prediction[0]:
        s_1 = 1
    else:
        s_1 = 0
    s1_list_10gram.append(s_1)

    # Check if the word appears in top 5 predictions to calculate s@5
    s_5 = 0
    for i in range(0, 4):
        if correct_word == _1gram_prediction[i]:
            s_5 = 1
            break
    s5_list_1gram.append(s_5)
    s_5 = 0
    for i in range(0, 4):
        if correct_word == _2gram_prediction[i]:
            s_5 = 1
            break
    s5_list_2gram.append(s_5)
    s_5 = 0
    for i in range(0, 4):
        if correct_word == _3gram_prediction[i]:
            s_5 = 1
            break
    s5_list_3gram.append(s_5)
    s_5 = 0
    for i in range(0, 4):
        if correct_word == _5gram_prediction[i]:
            s_5 = 1
            break
    s5_list_5gram.append(s_5)
    s_5 = 0
    for i in range(0, 4):
        if correct_word == _10gram_prediction[i]:
            s_5 = 1
            break
    s5_list_10gram.append(s_5)

    # Check if the word appears in top 10 predictions to calculate s@10
    s_10 = 0
    for i in range(0, 10):
        if correct_word == _1gram_prediction[i]:
            s_10 = 1
            break
    s10_list_1gram.append(s_10)
    s_10 = 0
    for i in range(0, 10):
        if correct_word == _2gram_prediction[i]:
            s_10 = 1
            break
    s10_list_2gram.append(s_10)
    s_10 = 0
    for i in range(0, 10):
        if correct_word == _3gram_prediction[i]:
            s_10 = 1
            break
    s10_list_3gram.append(s_10)
    s_10 = 0
    for i in range(0, 10):
        if correct_word == _5gram_prediction[i]:
            s_10 = 1
            break
    s10_list_5gram.append(s_10)
    s_10 = 0
    for i in range(0, 10):
        if correct_word == _10gram_prediction[i]:
            s_10 = 1
            break
    s10_list_10gram.append(s_10)

print("Average s@1 for 1 gram model ", pytrec_eval.compute_aggregated_measure("gm", s1_list_1gram))
print("Average s@1 for 2 gram model ", pytrec_eval.compute_aggregated_measure("gm", s1_list_2gram))
print("Average s@1 for 3 gram model ", pytrec_eval.compute_aggregated_measure("gm", s1_list_3gram))
print("Average s@1 for 5 gram model ", pytrec_eval.compute_aggregated_measure("gm", s1_list_5gram))
print("Average s@1 for 10 gram model ", pytrec_eval.compute_aggregated_measure("gm", s1_list_10gram))
print("Average s@5 for 1 gram model ", pytrec_eval.compute_aggregated_measure("gm", s5_list_1gram))
print("Average s@5 for 2 gram model ", pytrec_eval.compute_aggregated_measure("gm", s5_list_2gram))
print("Average s@5 for 3 gram model ", pytrec_eval.compute_aggregated_measure("gm", s5_list_3gram))
print("Average s@5 for 5 gram model ", pytrec_eval.compute_aggregated_measure("gm", s5_list_5gram))
print("Average s@5 for 10 gram model ", pytrec_eval.compute_aggregated_measure("gm", s5_list_10gram))
print("Average s@10 for 1 gram model ", pytrec_eval.compute_aggregated_measure("gm", s10_list_1gram))
print("Average s@10 for 2 gram model ", pytrec_eval.compute_aggregated_measure("gm", s10_list_2gram))
print("Average s@10 for 3 gram model ", pytrec_eval.compute_aggregated_measure("gm", s10_list_3gram))
print("Average s@10 for 5 gram model ", pytrec_eval.compute_aggregated_measure("gm", s10_list_5gram))
print("Average s@10 for 10 gram model ", pytrec_eval.compute_aggregated_measure("gm", s10_list_10gram))
