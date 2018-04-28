from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

# for w in example_words:
#     print(ps.stem(w))

new_text = "it is very important to be pythonly while you are pythoning with python. all pythoners have pythonly pythoned using a pythoner"

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))

# note: use wordnet with nltk and it will be stemmed in wordnet and synnet automatically