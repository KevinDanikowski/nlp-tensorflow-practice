# pickle for data files from signalmedia-1m.jsonl

import _pickle as pickle

heads = []
descs = []
keywords = []
# for filename in range(numOfFiles):
with open('../data/too-large-for-git/signalmedia-1m' + '.jsonl','rb') as fp:
    text = fp.readlines()[0:1000]
    temp = text
    heads.append(temp.split("\n")[0])
    descs.append(text) 
    keywords.append(None)
        
with open('../data/signalmedia_1000articles.pkl', 'wb') as f:
     pickle.dump((heads,descs,keywords),f)