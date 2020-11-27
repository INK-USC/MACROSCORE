import json, os, xml.etree.ElementTree as ET
from wordsegment import load, segment
from collections import Counter

load() # load word segmenter

data = json.load(open("./data/SCORE_json.json", 'r'))["data"]

all_xmls = os.listdir("./xml")

keys = ['claim2_abstract', 'claim2_pg', 'claim2_start', 'claim2_box', \
        'claim3a_concretehyp', 'claim3a_pg', 'claim3a_start', 'claim3a_box', \
        'claim3b_testspec', 'claim3b_pg', 'claim3b_start', 'claim3b_box', \
        'claim4_inftest', 'claim4_pg', 'claim4_start', 'claim4_box']

def vote_string(strings):
    aggregated = [[r[i] for r in strings] for i in range(len(strings[0]))]
    voted = [sorted(Counter(r).items(), key=lambda x:x[1], reverse=True)[0][0] for r in aggregated]
    #TODO

def strip_text(texts):
    heads = [r[:10] for r in texts]
    tails = [r[-10:] for r in texts]
    voted_head = [[r[i] for r in heads] for i in range(10)]
    #TODO

data_processed = {}
for d in data:
    filename = d['pdf_filename']
    filename = filename+'.xml' if filename[-4:] == '.pdf' else filename+'.pdf.xml'
    tree = ET.parse('./xml/'+filename)
    root = tree.getroot()
    all_ids, all_text = [], []
    for r in root[1][0]:
        id = r[0][0].attrib['id']
        text = r[0][0].text
        all_ids.append(id)
        all_text.append(text)
        # all_text.append([id, ' '.join(segment(text))])
    
    #TODO
    all_text = strip_text(all_text)
    
    data_processed[filename] = {'title': d['title_WOS'], 'authors': d['author_full'], \
                                'year': d['pub_year_WOS'], 'keywords': [d['keywords'], d['keywords_plus']], \
                                'text': [[all_ids[i], all_text[i]] for i in range(len(all_text))]}
    for k in keys:
        data_processed[filename][k] = d[k]

json.dump(data_processed, open("./data/data_processed.json", 'w'))

# write txt file for AutoPhrase
data_processed = json.load(open("./data/data_processed.json", 'r'))
with open("../AutoPhrase/data/papers/papers.txt", 'w') as f:
    for k, v in data_processed.items():
        for t in v['text']:
            if t[1]:
                f.write(t[1].replace('\n', '\\n')+'\n')



# use only sentences around/in claims
data_processed = json.load(open("./data/data_processed.json", 'r'))
window_size = 5
text = []
for k, v in data_processed.items():
    all_windows = ""
    for claim_name, claim_no in [['claim2_abstract', '2'], ['claim3a_concretehyp', '3a'], \
                                 ['claim3b_testspec', '3b'], ['claim4_inftest', '4']]:
        claim = v[claim_name].split(' | ')
        claim_pg = v['claim'+claim_no+'_pg'].split(' | ')
        claim_start = v['claim'+claim_no+'_start'].split(' | ')
        window_span = []
        if len(claim) != len(claim_pg):
            print("Split Error!")
            continue
        
        for i in range(len(claim)):
            pg = int(claim_pg[i]) - 1
            start = int(claim_start[i])
            end = start + len(claim[i])
            
            num_sents = -1
            i = start
            starts = []
            while i >= 0 and num_sents <= window_size:
                i -= 1
                if v['text'][pg][1][i] + v['text'][pg][1][i+1] == '. ':
                    num_sents += 1
                    starts.append(i)
            
            num_sents = -1
            i = end
            ends = []
            while i < len(v['text'][pg][1])-2 and num_sents <= window_size:
                i += 1
                if v['text'][pg][1][i] + v['text'][pg][1][i+1] == '. ':
                    num_sents += 1
                    ends.append(i)
            
            window_start = starts[-1] + 1 if starts else start
            window_end = ends[-1] + 2 if ends else end
            
            text.append(v['text'][pg][1][window_start:window_end])
            all_windows += ' ' + v['text'][pg][1][window_start:window_end]
    
    data_processed[k]['important_sents'] = all_windows

json.dump(data_processed, open("./data/data_processed.json", 'w'))


with open("../AutoPhrase/data/papers/papers.txt", 'w') as f:
    for t in text:
        f.write(t.replace('\n', '\\n')+'\n')


