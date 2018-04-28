import nltk
import random #because it's all first positive, then negative
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)] #tupuls

random.shuffle(documents)

all_words = [] 

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words) #ordered from most to least common words

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document) #all words
    features = {}
    for w in word_features:
        features[w] = (w in words) #boolean true / false, tells us if this review contains those top words
        # true only if the review contains one of top 3000 words

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

