from nltk.corpus import gutenberg #gutenberg bible
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

print(sent_tokenize(sample)[5:15])