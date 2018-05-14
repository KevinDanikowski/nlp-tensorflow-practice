import _pickle as pickle # cPickle
from collections import Counter #tallies the total count of words in a list
import keras
import postprocessing as pr #possibly from https://github.com/steerapi/seq2seq-show-att-tell/blob/master/generate_pretrained_embedding.py

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

#Create word embeddings with GloVe
path = '../data/too-large-for-git/glove.6B.zip'
glove_weights = get_glove_weights(path, origin="http://nlp.stanford.edu/data/glove.6B.zip")
word_embeddings = pr.build_glove_matrix(glove_weights, vocab)

#3 stacked LSTM RNN
def build_model(embedding):
    model = Sequential()
    model.add(Embedding(weights=[embedding], name='embedding_1'))
    for i in range(3):
        lstm = LSTM(rnn_size,
                    name='lstm_%d'%(i+1))
        model.add(lstm)
        model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))
    model.add(Dense())
    model.add(Activation('softmax', name='activation'))
    return model

# input data will be words from stories and 

#Initialize Encoder RNN with Embeddings
encoder = build_model(word_embeddings) #pretrained embeddings set to first layers weights, 
encoder.compile(loss='categorical_crossentropy', optimizer='rmsprop') #minimizes cross-entropy loss
encoder.save_weights('embeddings.pkl', overwrite=True)

#Initialize Decoder RNN with Embeddings
with open('embeddings.pkl', 'rb') as fp: # uses same pretrained glove embeddings
    embeddings = pickle.load(fp)
decoder = build_model(embedding)

#Convert a given article to a headline
headline1 = pr.gen_headline(decoder, desc[1])
print(headline1)