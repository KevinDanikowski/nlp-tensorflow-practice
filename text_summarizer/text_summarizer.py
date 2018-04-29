import _pickle # cPickle
from collections import Counter
import keras
# import postprocessing

# load data
with open('data/%s.pkl', 'rb') as fp:
    #heads, desc, and keywords as separate arrays
    heads, desc, keywords = pickle.load(fp)

# headings tupal
i = 0
heads[i]
# Remainders : Super wi-fi edition

#Articles
desc[i]

#tokenize text
def get_vocab(first):
    vocabcount, vocab = Counter(w for txt in first for w in txt.split())
    return vocab, vocabcount

vocab, vocabcount = get_vocab(head+desc)