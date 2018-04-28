from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
#lemmatizing groups words to their main stem kind of like synonnym
print(lemmatizer.lemmatize("better", pos="a")) #a = adjective, prints good because good vs better vs best has root good in meaning
# if no noun, must pass through parts of speech to be use
print(lemmatizer.lemmatize("run", 'v')) # prints run, v for verb
