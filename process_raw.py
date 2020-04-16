import os, re, json
from collections import defaultdict, Counter
from nltk.tokenize import sent_tokenize, word_tokenize

data_source = 'TA1'

if data_source == 'TA1':
    raw_dir = "./tagtog/tokenized"
    raw_files = os.listdir(raw_dir)
    raw_files = [r for r in raw_files if r != "0_Labels.txt"]

    metadata = json.load(open("./data/SCORE_json.json"))
    filename2id = {r['pdf_filename']:r['paper_id'] for r in metadata['data']}
    metadata = {r['pdf_filename']:r for r in metadata['data']}

    fout = open("./data_processed/raw_all.txt", 'w')
    data = []
    diffs = []
    for f in raw_files:
        content = open(raw_dir+'/'+f).readlines()
        content = [r[:-1] for r in content]
        content = content[3:]
        assert not '' in content and not '----NEW DOC----' in content
        filename = [r for r in filename2id if f.split('.')[0].split('_')[-1] in r]
        assert len(filename) == 1
        filename = filename[0]
        # as requested, extract features (effect sizes, p-values) around claim4
        claim4 = [r for r in metadata[filename]['claim4_inftest'].split(' | ') if r.count(' ') > 2]
        claim4_chars = [re.sub('\W', '', r.replace(" ", "")) for r in claim4]
        for i in range(len(content)):
            content_i_chars = re.sub('\W', '', content[i].replace(" ", ""))
            if any([r in content_i_chars for r in claim4_chars]):
                content[i] = "<<claim4>>" + content[i] # this sentence contains claim4
        
        fout.write('----NEW DOC----\n'+filename2id[filename]+'\n'+'\n'.join(content).lower() + '\n')

elif data_source == 'RPP':
    raw_dir = "./tagtog/tokenized_RPP"
    raw_files = os.listdir(raw_dir)
    raw_files = [r for r in raw_files if r != "0_Labels.txt"]

    filename2id = {r:r[:-4] for r in raw_files}

    fout = open("./data_processed/raw_all_RPP.txt", 'w')
    data = []
    for f in raw_files:
        content = open(raw_dir+'/'+f).readlines()
        content = [r[:-1] for r in content]
        content = content[3:]
        assert not '' in content and not '----NEW DOC----' in content
        filename = filename2id[f]
        
        fout.write('----NEW DOC----\n'+filename+'\n'+'\n'.join(content).lower() + '\n')

elif data_source == 'biomed':
    raw_dir = "./CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv"
    raw_files = os.listdir(raw_dir)
    
    fout = open("./data_processed/raw_all_biomed.txt", 'w')
    for f in raw_files:
        content = json.load(open(raw_dir+'/'+f, 'r'))
        paper_id = content['paper_id']
        fout.write('----NEW DOC----\n'+paper_id+'\n')
        for paragraph in content['body_text']:
            text = paragraph['text']
            cite_spans = [[r['start'], r['end']] for r in paragraph['cite_spans']]
            # combine adjacent cite spans
            if len(cite_spans)>1:
                prev_spans, all_spans = cite_spans[0], []
                for s in cite_spans[1:]:
                    if s[0] - prev_spans[1] <= 1:
                        prev_spans = [prev_spans[0], s[1]]
                    else:
                        all_spans.append(prev_spans)
                        prev_spans = s
                
                all_spans.append(prev_spans)
                cite_spans = all_spans
            
            # remove cites
            cite_spans.reverse()
            for s in cite_spans:
                text = text[:s[0]] + text[(s[1]+1):]
            
            sents = sent_tokenize(text)
            words = [word_tokenize(t) for t in sents]
            
            # fout.write(text.lower() + '\n')
            for s in words:
                fout.write(" ".join(s).lower() + '\n')


