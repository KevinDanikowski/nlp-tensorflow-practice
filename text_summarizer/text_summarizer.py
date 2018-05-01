import _pickle as pickle # cPickle
from collections import Counter
import keras
# import postprocessing

# load data
with open('../data/tokens.pkl', 'rb') as fp: # use create_picle_file.py
    #heads, desc, and keywords as separate arrays
    heads, descs, keywords = pickle.load(fp)

# headings tupal
i = 0
heads[i]
# Remainders : Super wi-fi edition

#Articles
descs[i]

#tokenize text
def get_vocab(first):
    vocabcount, vocab = Counter(w for txt in first for w in txt.split())
    print(vocabcount, vocab)
    #return vocab, vocabcount

vocab, vocabcount = get_vocab(heads[i]+descs[i])

#print (vocab[:50])
#print ('...', len(vocab))