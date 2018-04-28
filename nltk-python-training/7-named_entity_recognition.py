import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer #unsupervised ML, can retrain if you choose but is pretrained

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sentence_tokenizer = PunktSentenceTokenizer(train_text) #for training

tokenized = custom_sentence_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary=True) # types: organization, person, location, date, time, money, percent, facility. binary=true will classify it that way period instead of following identical matching type

            namedEnt.draw() # united states is one exmaple of NE



    except Exception as e:
        print(str(e))

process_content()