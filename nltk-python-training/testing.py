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
            print(tagged)
            # Example: [('We', 'PRP'), ('will', 'MD'), ('compete', 'VB'), ('and', 'CC'), ('excel', 'VB'), ('in', 'IN'),('the', 'DT'), ('global', 'JJ'), ('economy', 'NN'), ('.', '.')]
    except Exception as e:
        print(str(e))

# creates tupuls of words and parts of speech 
# tag list: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
process_content()