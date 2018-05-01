# pickle for data files from signalmedia-1m.jsonl
# info from https://github.com/llSourcell/How_to_make_a_text_summarizer/issues/11
import _pickle as pickle
import jsonlines

heads = []
descs = []
keywords = []

with jsonlines.open('../data/too-large-for-git/signalmedia-1m' + '.jsonl','r') as reader:
    i = 0
    for obj in reader:
        if i < 2:
            i += 1
            head = obj["title"]
            desc = obj["content"]
            heads.append(head)
            descs.append(desc)
            keywords.append(None)
        else:
            break
        
# print(heads, descs, keywords)
with open('../data/tokens.pkl', 'wb') as f:
    pickle.dump((heads,descs,keywords),f)