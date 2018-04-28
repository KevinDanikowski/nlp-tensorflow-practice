from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "this is an example sentence which we will use."
stop_words = set(stopwords.words("english"))

stop_word_results = {"you've", 'ourselves', "doesn't", 'be', 'in', 'which', 'here', 'will', 'when', 'shan', 'her', 'once', 'll', 'its', "hadn't", 'these', 'itself', 'aren', 'doing', "should've", 'during', "aren't", 'so', "wasn't", 'before', 'for', 'we', 'he', 'does', 'had', 'themselves', 'over', 'few', 'my', 'were', 'out', 'hasn', 'o', "won't", 'should', 'him', 'such', 'are', 'as', 'i', 'nor', 'how', 'below', 'other', 'herself', 'ma', "wouldn't", 'off', 'by', 'been', 'same', 'then', "you'd", 'the', 'all', 'a', 'that', 'on', 'hadn', "couldn't", 'your', "that'll", 'himself', 'yourselves', 'them','again', 'very', 'down', 'y', 's', 'if', 'can', 'have', 'at', 'no', 'didn', "don't", 're', 'this', 'they', 'don', 'than', 'because', 'through', 'it', 'is', 'most', 'couldn', 'of', "shan't", 'only', 'not', "weren't", 'weren', 'theirs', 'yourself', 'm', "shouldn't", "didn't", 'won', 'after','against', 'those', 'from', 'mustn', 'but', 'or', 'while', 'there', 'whom', 'up', 'me', 'to', 't', 'now', 'more', "she's", 'wouldn', "haven't", 'about', 'you', 'some', 'and', "isn't", 'hers', "needn't", "you'll", 'above', 'where', 'yours', 'wasn', 'both', 'do', 'too', 'being', 'haven', "you're", 'their', 'between', 'shouldn', 'ours', "it's", 'am', 'what', 'just', "hasn't", 'into', 'our', 'with', 'she', 'why', 'who', 'each', 'ain', "mightn't", 'did', 'own', 'has', 'an', 'd', 'doesn', 've', 'mightn', 'under', 'further', 'any', "mustn't", 'myself', 'his', 'having', 'needn', 'until', 'isn', 'was'}

words = word_tokenize(example_sentence)

filtered_sentence = []

for w in words: 
    if w not in stop_words:
        filtered_sentence.append(w)

# filtered_sentence = [w for w in words if not w in stop_words] is faster way to write
print(filtered_sentence) # result: ['example', 'sentence', 'use', '.']