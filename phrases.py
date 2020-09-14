import json, re, numpy as np
from collections import Counter
import spacy
from wordsegment import load, segment

load()
nlp = spacy.load("en_core_web_sm")


data_processed = json.load(open("./data/data_processed.json", 'r'))
phrased_text = open("../AutoPhrase/models/DBLP_papers/segmentation.txt").readlines()

def get_phrase_loc(raw, phrased):
    phrase_locs, curr_phrase_start = [], -1
    i, j = 0, 0
    in_phrase = False
    while i < len(raw):
        if phrased[j:(j+8)] == '<phrase>':
            assert not in_phrase
            j += 8
            in_phrase = True
            curr_phrase_start = i
        elif phrased[j:(j+9)] == '</phrase>':
            assert in_phrase
            j += 9
            phrase_locs.append([curr_phrase_start, i])
            in_phrase = False
        else:
            i += 1
            j += 1
    
    return phrase_locs

text_iter = iter(phrased_text)
for k in data_processed:
    for i in range(len(data_processed[k]['text'])):
        if data_processed[k]['text'][i][1]:
            phrase_locs = get_phrase_loc(data_processed[k]['text'][i][1], text_iter.__next__())
            data_processed[k]['text'][i].append(phrase_locs)

json.dump(data_processed, open("./data/data_with_phrases.json", 'w'))
data_with_phrases = json.load(open("./data/data_with_phrases.json", 'r'))

# phrases in/around claims
window_size = 500
phrases_in_claims = []
phrases_near_claims = []
for k, v in data_with_phrases.items():
    for claim_name, claim_no in [['claim2_abstract', '2'], ['claim3a_concretehyp', '3a'], \
                                 ['claim3b_testspec', '3b'], ['claim4_inftest', '4']]:
        claim = v[claim_name].split(' | ')
        claim_pg = v['claim'+claim_no+'_pg'].split(' | ')
        claim_start = v['claim'+claim_no+'_start'].split(' | ')
        claim_span, window_span = [], []
        if len(claim) != len(claim_pg):
            print("Split Error!")
            continue
        
        for i in range(len(claim)):
            pg = int(claim_pg[i])
            start = int(claim_start[i])
            end = start + len(claim[i])
            claim_span.append([pg-1, start, end])
            window_span.append([pg-1, start-window_size, start])
            window_span.append([pg-1, end, end+window_size])
        
        for r in claim_span:
            # v['text'][r[0]][1][r[1]:r[2]] #print span
            phrases_span = [rr for rr in v['text'][r[0]][2] if rr[0]>=r[1] and rr[1]<=r[2]]
            for s in phrases_span:
                phrases_in_claims.append(v['text'][r[0]][1][s[0]:s[1]].lower())
        
        for r in window_span:
            # v['text'][r[0]][1][r[1]:r[2]] #print span
            phrases_span = [rr for rr in v['text'][r[0]][2] if rr[0]>=r[1] and rr[1]<=r[2]]
            for s in phrases_span:
                phrases_near_claims.append(v['text'][r[0]][1][s[0]:s[1]].lower())

list(Counter(phrases_in_claims).keys())[:50]
list(Counter(phrases_near_claims).keys())[:50]
with open("phrases_in_claims.txt", 'w') as f:
    for p in list(Counter(phrases_in_claims).keys()):
        if ' ' in p or '-' in p:
            f.write(p+'\n')

with open("phrases_near_claims.txt", 'w') as f:
    for p in list(Counter(phrases_near_claims).keys()):
        if ' ' in p or '-' in p:
            f.write(p+'\n')


# TF-IDF around + claims
phrases = [r for r in set(phrases_in_claims + phrases_near_claims) if ' ' in r or '-' in r]
window_size = 500
phrases_in_claims = []
phrases_near_claims = []
for k, v in data_with_phrases.items():
    for claim_name, claim_no in [['claim2_abstract', '2'], ['claim3a_concretehyp', '3a'], \
                                 ['claim3b_testspec', '3b'], ['claim4_inftest', '4']]:
        claim = v[claim_name].split(' | ')
        claim_pg = v['claim'+claim_no+'_pg'].split(' | ')
        claim_start = v['claim'+claim_no+'_start'].split(' | ')
        claim_span, window_span = [], []
        if len(claim) != len(claim_pg):
            print("Split Error!")
            continue
        
        for i in range(len(claim)):
            pg = int(claim_pg[i])
            start = int(claim_start[i])
            end = start + len(claim[i])
            claim_span.append([pg-1, start, end])
            window_span.append([pg-1, start-window_size, start])
            window_span.append([pg-1, end, end+window_size])
        
        for r in claim_span:
            # v['text'][r[0]][1][r[1]:r[2]] #print span
            phrases_span = [rr for rr in v['text'][r[0]][2] if rr[0]>=r[1] and rr[1]<=r[2]]
            for s in phrases_span:
                phrases_in_claims.append(v['text'][r[0]][1][s[0]:s[1]].lower())
        
        for r in window_span:
            # v['text'][r[0]][1][r[1]:r[2]] #print span
            phrases_span = [rr for rr in v['text'][r[0]][2] if rr[0]>=r[1] and rr[1]<=r[2]]
            for s in phrases_span:
                phrases_near_claims.append(v['text'][r[0]][1][s[0]:s[1]].lower())



v_miss_count, i_miss_count, d_miss_count, id_miss_count, ov_count, c_count, s_miss_count = 0, 0, 0, 0, 0, 0, 0
for v in data_with_phrases.values():
    text = ''.join([r[1] for r in v['text'] if r[1]]).lower()
    if not re.match('.*varia.*', text):
        v_miss_count += 1
    if not re.match('.*independent\s*variable.*', text):
        i_miss_count += 1
    m = re.findall('..dependent\s*variable', text)
    has_d = False
    for mm in m:
        if not 'independent' in mm:
            has_d = True
            break
    
    if not has_d:
        d_miss_count += 1
    
    if not re.match('.*independent\s*variable.*', text) and not has_d:
        id_miss_count += 1
        if re.match('.*covaria.*', text):
            c_count += 1
        elif re.match('.*varia.*', text):
            i = text.index('varia')
            print(text[i-30:i+30])
            ov_count += 1
    
    if not re.match('.*significan.*', text) and not re.match('.*statistic.*', text):
        s_miss_count += 1

i_miss_count, d_miss_count, s_miss_count



# check keywords
keywords = open('keywords.txt', 'r').readlines()
keywords = [r.strip().lower().split(', ') for r in keywords]
keywords = {tuple(k):0 for k in keywords}
for k, v in data_with_phrases.items():
    text = ''.join([p[1].lower() for p in v['text'] if p[1]])
    for key in keywords:
        for key_ in key:
            if key_ in text:
                keywords[key] += 1
                break

for k, v in keywords.items():
    print(k, '\t', v)

# mutual information
contrary_keywords = open('contrary_keywords.txt', 'r').readlines()
contrary_keywords = [[rr.split(',') for rr in r.strip().lower().split(' ||| ')] for r in contrary_keywords]
contrary_keywords = {tuple(k):0 for k in contrary_keywords}
for pair in contrary_keywords:
    p0, p1, p01 = 0, 0, 0
    for k, v in data_with_phrases.items():
        text = ''.join([p[1].lower() for p in v['text'] if p[1]])
        flag0, flag1 = 0, 0
        for key in pair[0]:
            if key in text:
                p0 += 1
                flag0 = 1
                break
        
        for key in pair[1]:
            if key in text:
                p1 += 1
                flag1 = 1
                break
        
        if flag0 and flag1:
            p01 += 1
    
    p0 /= len(data_with_phrases)
    p1 /= len(data_with_phrases)
    p01 /= len(data_with_phrases)
    MI = p01 / (p0 * p1)
    log_MI = np.log(MI)
    print(pair, '\t', MI, '\t', log_MI)


# Text with phrases for Daniel
data_with_phrases = json.load(open("./data/data_with_phrases.json", 'r'))
window_size = 5 # number of sentences before/after claims
results = [[], [], [], []]
for k, v in data_with_phrases.items():
    if '|' in v['claim2_pg'] + v['claim3a_pg'] + v['claim3b_pg'] + v['claim4_pg'] + \
              v['claim2_abstract'] + v['claim3a_concretehyp'] + v['claim3b_testspec'] + \
              v['claim4_inftest']:
        continue
    
    for claim_name, claim_no in [['claim2_abstract', '2'], ['claim3a_concretehyp', '3a'], \
                                 ['claim3b_testspec', '3b'], ['claim4_inftest', '4']]:
        claim = v[claim_name]
        claim_pg = v['claim'+claim_no+'_pg']
        claim_start = v['claim'+claim_no+'_start']
        
        pg = int(claim_pg) - 1
        start = int(claim_start)
        end = start + len(claim) + 1
        claim_span = [pg, start, end]
        
        break_flag = False
        num_sents = -1
        i = start
        starts = []
        while num_sents <= window_size:
            i -= 1
            if i <= 0:
                break_flag = True
                break
            
            if v['text'][pg][1][i] + v['text'][pg][1][i+1] == '. ':
                num_sents += 1
                starts.append(i)
        
        if break_flag:
            break
        
        window_start = starts[-1] + 1
        
        num_sents = -1
        i = end
        ends = []
        while num_sents <= window_size:
            i += 1
            if i >= len(v['text'][pg][1])-1:
                break_flag = True
                break
            
            if v['text'][pg][1][i] + v['text'][pg][1][i+1] == '. ':
                num_sents += 1
                ends.append(i)
        
        if break_flag:
            break
        
        window_end = ends[-1] + 2
        
        before_span = [pg, window_start, start]
        after_span = [pg, end, window_end]
        
        before_phrases = [r for r in v['text'][pg][2] if before_span[1] <= r[0] and r[1] < before_span[2]]
        after_phrases = [r for r in v['text'][pg][2] if after_span[1] <= r[0] and r[1] < after_span[2]]
        claim_phrases = [r for r in v['text'][pg][2] if start <= r[0] and r[1] < end]
        
        before = v['text'][before_span[0]][1][before_span[1]:before_span[2]]
        after = v['text'][after_span[0]][1][after_span[1]:after_span[2]]
        claim = v['text'][claim_span[0]][1][claim_span[1]:claim_span[2]]
        
        i = before_span[1]
        for r in before_phrases:
            # if not ' ' in before[(r[0]-i):(r[1]-i)] and not '-' in before[(r[0]-i):(r[1]-i)]:
                # continue
            
            if not before[(r[0]-i)-1] == before[(r[1]-i)] == ' ':
                continue
            
            before = before[:(r[0]-i)] + '{' + before[(r[0]-i):]
            i -= 1
            before = before[:(r[1]-i)] + '}' + before[(r[1]-i):]
            i -= 1
        
        i = after_span[1]
        for r in after_phrases:
            # if not ' ' in after[(r[0]-i):(r[1]-i)] and not '-' in after[(r[0]-i):(r[1]-i)]:
                # continue
            
            if not after[(r[0]-i)-1] == after[(r[1]-i)] == ' ':
                continue
            
            after = after[:(r[0]-i)] + '{' + after[(r[0]-i):]
            i -= 1
            after = after[:(r[1]-i)] + '}' + after[(r[1]-i):]
            i -= 1
        
        i = start
        for r in claim_phrases:
            # if not ' ' in claim[(r[0]-i):(r[1]-i)] and not '-' in claim[(r[0]-i):(r[1]-i)]:
                # continue
            
            if not claim[(r[0]-i)-1] == claim[(r[1]-i)] == ' ':
                continue
            
            claim = claim[:(r[0]-i)] + '{' + claim[(r[0]-i):]
            i -= 1
            claim = claim[:(r[1]-i)] + '}' + claim[(r[1]-i):]
            i -= 1
        
        before = nlp(before)
        after = nlp(after)
        
        # before = nlp(v['text'][before_span[0]][1][before_span[1]:before_span[2]])
        # claim = v['text'][claim_span[0]][1][claim_span[1]:claim_span[2]]
        # after = nlp(v['text'][after_span[0]][1][after_span[1]:after_span[2]])
        
        before = list(before.sents)[-(window_size):]
        after = list(after.sents)[0:(window_size)]
        
        before = ' '.join([r.text for r in before])
        after = ' '.join([r.text for r in after])
        
        print(before+"\n")
        print(claim+"\n")
        print(after+"\n\n\n")
        
        if claim_no == '2':
            results[0].append([before, claim, after])
        elif claim_no == '3a':
            results[1].append([before, claim, after])
        elif claim_no == '3b':
            results[2].append([before, claim, after])
        else:
            results[3].append([before, claim, after])
        

with open('output_for_dan.txt', 'w') as f:
    for i, claim_name in enumerate(['Claim 1: Abstract', 'Claim 2: Concrete Hypotheses', \
                                    'Claim 3: Test Specification', 'Claim 4: Inferential Test']):
        f.write("\n\n" + "*"*60 + "\n" + claim_name + "\n" + "*"*60 + "\n\n")
        for j in range(10):
            before, claim, after = results[i][j]
            f.write(str(i+1)+'.'+str(j+1) + '\n>    ' + before + '\n>    ' + claim + '\n>    ' + after + '\n\n')



# TF-IDF of phrases in important sentences (in/around claims)
data_processed = json.load(open("./data/data_processed.json", 'r'))
phrased_text = open("../AutoPhrase/models/DBLP_papers/segmentation.txt").readlines()
phrases = open("../AutoPhrase/models/DBLP_papers/AutoPhrase.txt").readlines()
phrases_ = []
for r in phrases:
    split = r.strip().split('\t')
    if len(split) > 1:
        if float(split[0]) > 0.7:
            if any([split[1] in r['important_sents'] for r in data_processed.values()]):
                phrases_.append(split[1])

phrases = phrases_

data = json.load(open("./data/SCORE_json.json", 'r'))["data"]
sorted_files = []
for d in data:
    filename = d['pdf_filename']
    filename = filename+'.xml' if filename[-4:] == '.pdf' else filename+'.pdf.xml'
    assert filename in data_processed
    sorted_files.append(filename)

tf = []
for f in sorted_files:
    curr_tf = []
    for p in phrases:
        curr_tf.append(data_processed[f]['important_sents'].count(p))
    
    tf.append(curr_tf)

df = []
for p in phrases:
    curr_df = 0
    for f in sorted_files:
        if p in data_processed[f]['important_sents']:
            curr_df += 1
    
    df.append(curr_df)

idf = [np.log(len(sorted_files) / r) for r in df]

tfidf = [[r[i]*idf[i] for i in range(len(r))] for r in tf]

open("tf.csv", 'w').write('\n'.join([','.join([str(rr) for rr in r]) for r in tf]))
open("papers.txt", 'w').write('\n'.join(sorted_files))
open("phrases_and_df.tsv", 'w').write('\n'.join([r[0]+'\t'+str(r[1]) for r in zip(phrases, df)]))
open("tf_idf.csv", 'w').write('\n'.join([','.join(["%.3f" % rr for rr in r]) for r in tfidf]))



