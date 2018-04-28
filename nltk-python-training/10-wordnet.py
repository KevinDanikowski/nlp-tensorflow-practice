from nltk.corpus import wordnet #takes some time to import

syns = wordnet.synsets("program")

#print(syns) #[Synset('plan.n.01'), Synset('program.n.02'), Synset('broadcast.n.02'), Synset('platform.n.02'),Synset('program.n.05'), Synset('course_of_study.n.01'), Synset('program.n.07'), Synset('program.n.08'), Synset('program.v.01'), Synset('program.v.02')]

#print(syns[0].lemmas()) #what versions of the first synonym it has

# print(syns[0].definition())
# print(syns[0].examples())
# print(syns[0].name())

synonyms=[]
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2)) #wu and palmer = semantic symilarity between words, returns a number