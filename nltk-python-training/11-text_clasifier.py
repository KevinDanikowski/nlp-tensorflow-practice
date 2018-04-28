import nltk
import random #because it's all first positive, then negative
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)] #tupuls

random.shuffle(documents)

# print(documents[1]) # list of words

all_words = [] #50k words or more

for w in movie_reviews.words():
    all_words.append(w.lower())


all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15)) #printing common words, most are punctuation though

# print(all_words["stupid"]) # prints 253 because 253 times stupid is said
