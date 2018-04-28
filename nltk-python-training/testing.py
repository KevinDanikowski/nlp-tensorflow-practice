import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer #unsupervised ML, can retrain if you choose but is pretrained

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sentence_tokenizer = PunktSentenceTokenizer(train_text) #for training

tokenized = custom_sentence_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram  = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?} """ # r """""" show regex. < starts find, RB for adverb, . for anything, ? for 0-1 of them, * for any number of it. <NNP> is required as you can see from no * after it. Chunk: just notes what you call it

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            #print(chunked) # result: finding all these nouns: (S   And/CC   so/RB   we/PRP   move/VBP   forward/RB   --/:   optimistic/JJ   about/IN   our/PRP$   country/NN   ,/,   faithful/JJ   to/TO   its/PRP$   cause/NN   ,/,   and/CC   confident/NN   of/IN   the/DT   victories/NNS   to/TO   come/VB   ./.) 
            chunked.draw()







    except Exception as e:
        print(str(e))

# creates tupuls of words and parts of speech 
# tag list: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
process_content()