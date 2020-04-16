import json
from collections import defaultdict

data = json.load(open("./data_processed/data.json", 'r'))

all_spans = defaultdict(lambda: [])

for d in data:
    all_words = [r for rr in d['sents'] for r in rr]
    for t in d['tags']:
        span = all_words[t[1][0]:t[2][0]]
        all_spans[t[0]].append(span)


for d in data:
    all_words = [r for rr in d['sents'] for r in rr]
    for t in [r for r in d['tags'] if r[0]=='SD']:
        span = all_words[t[1][0]:t[2][0]]
        if len(span) > 5:
            print(d['file'])
            print(span)
            print()

