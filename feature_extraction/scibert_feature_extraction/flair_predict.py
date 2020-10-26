import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import argparse, re, numpy as np, json
from collections import defaultdict
from transformers import AutoConfig, AutoModel
from tqdm import tqdm

# parser = argparse.ArgumentParser(description="Train flair")
# parser.add_argument("--folder", type=str, help="folder to chkp")
# args = parser.parse_args()
# args = vars(args)
# tagger = SequenceTagger.load('./flair_models/'+args['folder']+'/final-model.pt')

metadata = json.load(open("../../data/SCORE_json.json"))
id2doi = {r['paper_id']: r['DOI_CR'] for r in metadata['data']}
rppmap = json.load(open("../../data/doi_to_file_name_data.json"))
rppid2doi = {r['file']: r['doi'] for r in rppmap}
print(rppid2doi)
model_num = '1'

tagger = SequenceTagger.load('./model/final-model.pt') # load to gpu:0 by default, use CUDA_VISIBLE_DEVICES=x
# get name of embedding
transformer_model_name = '-'.join(tagger.embeddings.name.split('-')[2:])
print(transformer_model_name)

# reload transformer embedding
config = AutoConfig.from_pretrained(transformer_model_name, output_hidden_states=True)
tagger.embeddings.model = AutoModel.from_pretrained(transformer_model_name, config=config)
tagger.to(flair.device)

data_source = 'TA1'
if data_source == 'TA1':
    postfix = ''
elif data_source == 'RPP':
    postfix = '_RPP'
elif data_source == 'biomed':
    postfix = '_biomed'

data, curr_data, curr_id = {}, [], None
with open("../../data_processed/raw_all"+postfix+".txt", 'r') as f:
    f.readline()
    curr_id = f.readline()[:-1]
    while True:
        line = f.readline()
        if line in ["", "----NEW DOC----\n"]:
            assert curr_data
            data[curr_id] = curr_data
            curr_data = []
            if line == "":
                break
            curr_id = f.readline()[:-1]
        else:
            curr_data.append(line[:-1])

all_pred = {}
for id, sents in tqdm(data.items()):
    curr_pred = defaultdict(lambda: [])
    claim2_sents_idx = []
    claim3a_sents_idx = []
    claim3b_sents_idx = []
    claim4_sents_idx = []
    for i in range(len(sents)):
        sent = sents[i]
        # for TA1 only: store sent idx that contain claim4
        if data_source == 'TA1':
            if sent.startswith("<<claim2>>"):
                sent = sent[10:]
                claim2_sents_idx.append(i)
            if sent.startswith("<<claim3a>>"):
                sent = sent[11:]
                claim3a_sents_idx.append(i)
            if sent.startswith("<<claim3b>>"):
                sent = sent[11:]
                claim3b_sents_idx.append(i)
            if sent.startswith("<<claim4>>"):
                sent = sent[10:]
                claim4_sents_idx.append(i)
        sent = sent.split(' ')
        sentence = Sentence(' '.join(sent))
        if len(sentence.tokens) > 128 or len(sentence.tokens) <= 1 or sentence.tokens[-1].text not in ['.']:
            continue
        _ = tagger.predict(sentence)
        pred_spans = sentence.get_spans('ner')
        # for TA1 only: store sent idx that contain effect sizes or p-values
        for s in pred_spans:
            if data_source == 'TA1' and s.tag in ['ES', 'PV']:
                curr_pred[s.tag].append([s.text, i])
            else:
                curr_pred[s.tag].append(s.text)
    
    # for TA1 only: calculate distance between the sentence and nearest claim4
    curr_pred = dict(curr_pred)
    if data_source == 'TA1':
        for k in ['ES', 'PV']:
            if not k in curr_pred:
                continue
            for i in range(len(curr_pred[k])):
                if not claim4_sents_idx:
                    curr_pred[k][i][1] = None # if no claim4 found, give None
                else:
                    sent_idx = curr_pred[k][i][1]
                    curr_pred[k][i][1] = sorted([sent_idx - r for r in claim4_sents_idx], key=lambda x:abs(x))[0]

    all_pred[id] = curr_pred
    

# filter & normalize predictions
word2number = {'zero':0,'a':1,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,\
               'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,\
               'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,\
               'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,'thousand':1000,'million':1000000}

def text2number(text):
    l = re.split('-| ', text)
    base = None
    for r in l:
        if r in ['hundred', 'thousand', 'million']:
            if base == None:
                base = 1
            base *= word2number[r]
        elif r in word2number:
            if base == None:
                base = word2number[r]
            else:
                base += word2number[r]
        elif r == 'and':
            if base == None:
                return None
        else:
            # if a number exists, return it. It not detected any number, continue (may be some leading trivial words)
            if base:
                return base
    
    return base

def read_list_numbers(l, min_, max_):
    result, counter = [], 0
    for r in l:
        if r == ',':
            continue
        try:
            number = float(r)
            if number >= max_ or number <= min_:
                continue
            result.append(number)
        except:
            try:
                r1 = re.findall('[-\.0-9]+', r)[0]
                r2 = '.'.join(r1.split('.')[:2])
                number = float(r2)
                if number >= max_ or number <= min_:
                    continue
                result.append(number)
            except:
                counter += 1
                if counter > 1:
                    break
    
    return result

def process_pred(type, pred):
    pred = pred.strip()
    pred = pred.replace('−', '-')
    result = []
    if type == 'ES':
        pred_split = pred.split(' ')
        if len(pred_split) < 2:
            return None
        if pred_split[0] == 'r':
            if pred_split[1] == '2':
                if len(pred_split) < 3:
                    return None
                tag = 'R2'
                for i in range(2, min(len(pred_split), 10)):
                    try:
                        number = float(re.findall('[-\.0-9]+', pred_split[i])[0])
                        if 0 < number < 1:
                            result = read_list_numbers(pred_split[i:], 0, 1)
                            break
                    except:
                        continue
            elif pred_split[1] == ')':
                return None
            elif pred_split[1] == '(':
                tag = 'r'
                for i in range(2, min(len(pred_split), 6)):
                    try:
                        number = float(re.findall('[-\.0-9]+', pred_split[i])[0])
                        if -1 < number < 1:
                            result = read_list_numbers(pred_split[i:], -1, 1)
                            break
                    except:
                        continue
            else:
                tag = 'r'
                for i in range(0, min(len(pred_split), 6)):
                    try:
                        number = float(re.findall('[-\.0-9]+', pred_split[i])[0])
                        if -1 < number < 1:
                            result = read_list_numbers(pred_split[i:], -1, 1)
                            break
                    except:
                        continue
        elif 'r2' in pred_split[0] or 'r-squared' in pred_split[0] or pred_split[:2] == ['adjusted', 'r2']:
            tag = 'R2'
            for i in range(0, min(len(pred_split), 6)):
                try:
                    number = float(re.findall('[-\.0-9]+', pred_split[i])[0])
                    if 0 < number < 1:
                        result = read_list_numbers(pred_split[i:], 0, 1)
                        break
                except:
                    continue
        
        else:
            return None
        
        if result:
            return tag, result
    
    elif type == 'PV':
        pred_split = pred.split(' ')
        if '*' in pred_split[0]:
            return None
        if 'p' in pred_split[0] and (len(pred_split[0]) < 3 or re.findall('[-\.0-9]+', pred_split[0])):
            tag = 'p'
            for i in range(0, min(len(pred_split), 5)):
                try:
                    r1 = re.findall('[-\.0-9]+', pred_split[i])[0]
                    r2 = '.'.join(r1.split('.')[:2])
                    number = float(r2)
                    if 0 < number < 1:
                        result = read_list_numbers(pred_split[i:], 0, 1)
                        break
                except:
                    continue
        
        if result:
            return tag, result
    
    elif type == 'SS':
        pred = pred.replace(',', '')
        pred_split = pred.split(' ')
        try:
            number = int(pred_split[0])
            return number
        except:
            pass
        
        # number = text2number(pred_split[0])
        number = text2number(pred)
        if number:
            return number
    
    elif type in ['SD', 'TE']:
        name = pred.split(' ')[0]
        if not ('stud' in name or 'experiment' in name or 'model' in name or 'result' in name):
            return None
        numbers = [int(r) for r in re.findall('\d+', pred)]
        numbers = [n for n in numbers if n < 15]
        if numbers:
            return max(numbers)
    
    elif type == 'TN':
        pred = pred.replace('- ', '')
        return pred
    
    return None


output = {}
for id, preds in all_pred.items():
    curr_output = {}
    
    if 'SS' in preds:
        processed = [process_pred('SS', r) for r in preds['SS']]
        processed = [r for r in processed if r]
        curr_output['Sample Sizes'] = list(set(processed))
    else:
        curr_output['Sample Sizes'] = None
    
    if 'TN' in preds:
        processed = [process_pred('TN', r) for r in preds['TN']]
        processed = set([r for r in processed if r])
        processed = sorted(processed, key=lambda x:len(x), reverse=True)
        processed = [[r, set(re.findall('[^-^\s]+', r))] for r in processed]
        for i in range(1, len(processed)):
            for j in range(0,i):
                if processed[j] is None:
                    continue
                if processed[i][1].issubset(processed[j][1]):
                    processed[i] = None
                    break
        
        processed = [r[0] for r in processed if r]
        curr_output['Model Names'] = processed
    else:
        curr_output['Model Names'] = None
    
    if 'TE' in preds:
        processed = [process_pred('TE', r) for r in preds['TE']]
        processed = [r for r in processed if r]
        if processed:
            curr_output['Number of Models/Tests'] = max(processed)
        else:
            curr_output['Number of Models/Tests'] = None
    else:
        curr_output['Number of Models/Tests'] = None
    
    if 'SD' in preds:
        processed = [process_pred('SD', r) for r in preds['SD']]
        processed = [r for r in processed if r]
        if processed:
            curr_output['Number of Studies'] = max(processed)
        else:
            curr_output['Number of Studies'] = None
    else:
        curr_output['Number of Studies'] = None
    
    if 'PV' in preds:
        if data_source == 'TA1':
            processed = [[process_pred('PV', r[0]), r[1]] for r in preds['PV']]
            processed = [r for r in processed if r[0]]
            processed = [[rr, r[1]] for r in processed for rr in r[0][1]]
            processed = sorted(processed, key=lambda x:(abs(x[1]) if not x[1] is None else 1e9)) # sort on distance from nearest claim4 sent
        else:
            processed = [process_pred('PV', r) for r in preds['PV']]
            processed = [r for r in processed if r]
            processed = [r for rr in processed for r in rr[1]]
        curr_output['P Values'] = processed
    else:
        curr_output['P Values'] = None
    
    if 'ES' in preds:
        if data_source == 'TA1':
            processed = [[process_pred('ES', r[0]), r[1]] for r in preds['ES']]
            processed = [r for r in processed if r[0]]
            processed = [[xx, x[1]] for x in processed for xx in x[0][1]]
            processed = sorted(processed, key=lambda x: (abs(x[1]) if not x[1] is None else 1e9))

            r = [x for x in processed if x[0][0] == 'r']
            R2 = [x for x in processed if x[0][0] == 'R2']
            r = [[xx, x[1]] for x in r for xx in x[0][1]]
            r = sorted(r, key=lambda x:(abs(x[1]) if not x[1] is None else 1e9))
            R2 = [[xx, x[1]] for x in R2 for xx in x[0][1]]
            R2 = sorted(R2, key=lambda x:(abs(x[1]) if not x[1] is None else 1e9))
        else:
            processed = [process_pred('ES', r) for r in preds['ES']]
            processed = [r for r in processed if r]
            processed = [r for rr in processed for r in rr[1]]

        curr_output['Effect Sizes'] = processed
    else:
        curr_output['Effect Sizes'] = None

    if data_source == 'TA1':
        id = id2doi[id]
    elif data_source == 'RPP':
        id = rppid2doi[id]

    output[id] = curr_output


# inspect final results
# id = '8wZ0'
# for r in output[id].items():
    # print(r)

# for r in all_pred[id].items():
    # print(r)


# write to file
if data_source == 'TA1':
    postfix = "_with_claim4"

with open("./flair_pred/extraction_result_"+model_num+postfix+".csv", 'w') as f:
    f.write('Paper ID,Sample Sizes,Model Names,Number of Models/Tests,'
            'Number of Studies,P Values,Effect Sizes - r,Effect Sizes - R2\n')
    for id, out in output.items():
        SS = str(out['Sample Sizes']) if out['Sample Sizes'] else 'None'
        TN = str(out['Model Names']).replace('"', '') if out['Model Names'] else 'None'
        # TN = TN.replace(',', '')
        TE = str(out['Number of Models/Tests']) if out['Number of Models/Tests'] else 'None'
        SD = str(out['Number of Studies']) if out['Number of Studies'] else 'None'
        PV = str(out['P Values']) if out['P Values'] else 'None'
        ESr = str(out['Effect Sizes']) if out['Effect Sizes'] else 'None'
        ESR2 = str(out['Effect Sizes']) if out['Effect Sizes'] else 'None'
        f.write('"'+id+'","'+SS+'","'+TN+'","'+TE+'","'+SD+'","'+PV+'","'+ESr+'","'+ESR2+'"\n')

json.dump(output, open("./flair_pred/extraction_result_"+model_num+postfix+".parsed_rpp", 'w'))

