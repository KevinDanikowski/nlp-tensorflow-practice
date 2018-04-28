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

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900] #word true if shows up, category for positive / negative to train sentiment
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo Accuracy Percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# OUTPUT:
# Naive Bayes Algo Accuracy Percent:  82.0
# Most Informative Features
#               schumacher = True              neg : pos    =     11.4 : 1.0
#                    sucks = True              neg : pos    =      9.6 : 1.0
#                   justin = True              neg : pos    =      9.5 : 1.0
#                   annual = True              pos : neg    =      9.2 : 1.0
#                  frances = True              pos : neg    =      9.2 : 1.0
#            unimaginative = True              neg : pos    =      8.2 : 1.0
#              silverstone = True              neg : pos    =      7.5 : 1.0
#                  idiotic = True              neg : pos    =      7.3 : 1.0
#                   shoddy = True              neg : pos    =      6.9 : 1.0
#                  singers = True              pos : neg    =      6.5 : 1.0
#                  cunning = True              pos : neg    =      6.5 : 1.0
#                atrocious = True              neg : pos    =      6.5 : 1.0
#                   regard = True              pos : neg    =      6.3 : 1.0
#                   suvari = True              neg : pos    =      6.2 : 1.0
#                     mena = True              neg : pos    =      6.2 : 1.0