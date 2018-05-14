import _pickle as pickle # cPickle
from collections import Counter #tallies the total count of words in a list
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

#tokenize text, return vocab in order of usage (the should come first)
def get_vocab(combinedText):
    # TODO try to get vocab and count in another way, 
    words = combinedText.split()
    vocab = [word for word, word_count in Counter(words).most_common()]
    return vocab

vocab = get_vocab(heads[i]+descs[i])

print (vocab[:50])
print ('...', len(vocab))