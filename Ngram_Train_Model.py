import nltk
from nltk.util import ngrams
from nltk.corpus import brown
from nltk.corpus import stopwords
import pickle


nltk.data.path.append('./NLTK_Data/')
stop_words = set(stopwords.words('english'))


def remove_stopwords(x):
    y = []
    for pair in x:
        count = 0
        for word in pair:
            if word in stop_words:
                count = count or 0
            else:
                count = count or 1
        if (count == 1):
            y.append(pair)
    return (y)


print("Models training started")
# brown_news_corpus = brown.sents(categories=['news'])
brown_corpus_words = brown.words(categories='news')
brown_corpus_words = list(brown_corpus_words)

_1gram = []
_2gram = []
_3gram = []
_5gram = []
_10gram = []

print("Preprocessing Corpus")
print("Removing Punctuation and stopwords")
for word in brown_corpus_words:
    if word == '.' or word == ',' or word == '?' or word == '!':
        brown_corpus_words.remove(word)

_1gram.extend(list(ngrams(brown_corpus_words, 2)))
_2gram.extend(list(ngrams(brown_corpus_words, 3)))
_3gram.extend(list(ngrams(brown_corpus_words, 4)))
_5gram.extend(list(ngrams(brown_corpus_words, 6)))
_10gram.extend(list(ngrams(brown_corpus_words, 11)))

_1gram = remove_stopwords(_1gram)
_2gram = remove_stopwords(_2gram)
_3gram = remove_stopwords(_3gram)
_5gram = remove_stopwords(_5gram)
_10gram = remove_stopwords(_10gram)

freq_1gram = nltk.FreqDist(_1gram)
freq_2gram = nltk.FreqDist(_2gram)
freq_3gram = nltk.FreqDist(_3gram)
freq_5gram = nltk.FreqDist(_5gram)
freq_10gram = nltk.FreqDist(_10gram)

_1gram_voc = set(_1gram)
_2gram_voc = set(_2gram)
_3gram_voc = set(_3gram)
_5gram_voc = set(_5gram)
_10gram_voc = set(_10gram)

total_1gram = len(_1gram)
total_2gram = len(_2gram)
total_3gram = len(_3gram)
total_5gram = len(_5gram)
total_10gram = len(_10gram)

total_vocab_1gram = len(_1gram_voc)
total_vocab_2gram = len(_2gram_voc)
total_vocab_3gram = len(_3gram_voc)
total_vocab_5gram = len(_5gram_voc)
total_vocab_10gram = len(_10gram_voc)

_1gram_prob = []
_2gram_prob = []
_3gram_prob = []
_5gram_prob = []
_10gram_prob = []

print("Calculating Probabilities 1 gram model")
for ngram1 in _1gram_voc:
    _1list = [ngram1, _1gram.count(ngram1)]
    _1gram_prob.append(_1list)

for ngram1 in _1gram_prob:
    ngram1[-1] = (ngram1[-1] + 1) / (total_1gram + total_vocab_1gram)

print("Calculating Probabilities 2 gram model")
for ngram2 in _2gram_voc:
    _2list = [ngram2, _2gram.count(ngram2)]
    _2gram_prob.append(_2list)

for ngram2 in _2gram_prob:
    ngram2[-1] = (ngram2[-1] + 1) / (total_2gram + total_vocab_2gram)

print("Calculating Probabilities 3 gram model")
for ngram3 in _3gram_voc:
    _3list = [ngram3, _3gram.count(ngram3)]
    _3gram_prob.append(_3list)

for ngram3 in _3gram_prob:
    ngram3[-1] = (ngram3[-1] + 1) / (total_3gram + total_vocab_3gram)

print("Calculating Probabilities 5 gram model")
for ngram5 in _5gram_voc:
    _5list = [ngram5, _5gram.count(ngram5)]
    _5gram_prob.append(_5list)

for ngram5 in _5gram_prob:
    ngram5[-1] = (ngram5[-1] + 1) / (total_5gram + total_vocab_5gram)

print("Calculating Probabilities 10 gram model")
for ngram10 in _10gram_voc:
    _10list = [ngram10, _10gram.count(ngram10)]
    _10gram_prob.append(_10list)

for ngram10 in _10gram_prob:
    ngram10[-1] = (ngram10[-1] + 1) / (total_10gram + total_vocab_10gram)


# Sorting probability list in descending order
_1gram_prob = sorted(_1gram_prob, key=lambda l: l[1], reverse=True)
_2gram_prob = sorted(_2gram_prob, key=lambda l: l[1], reverse=True)
_3gram_prob = sorted(_3gram_prob, key=lambda l: l[1], reverse=True)
_5gram_prob = sorted(_5gram_prob, key=lambda l: l[1], reverse=True)
_10gram_prob = sorted(_10gram_prob, key=lambda l: l[1], reverse=True)

# Saving the trained model probabilities
pickle.dump(_1gram_prob, open('1-Gram-Model.pkl', 'wb'))
pickle.dump(_2gram_prob, open('2-Gram-Model.pkl', 'wb'))
pickle.dump(_3gram_prob, open('3-Gram-Model.pkl', 'wb'))
pickle.dump(_5gram_prob, open('5-Gram-Model.pkl', 'wb'))
pickle.dump(_10gram_prob, open('10-Gram-Model.pkl', 'wb'))











