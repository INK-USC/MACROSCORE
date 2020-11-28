import flair, torch
from flair.data import Sentence
from flair.models import SequenceTagger
import argparse, re, json, numpy as np
from collections import defaultdict
from transformers import AutoConfig, AutoModel
from tqdm import tqdm
import warnings
import nltk
import os
import sys

# Ignore the warnings
warnings.filterwarnings("ignore")

# Filter & Normalize the predictions
def text2number(text):
    word2number = {'zero': 0, 'a': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                   'eight': 8, 'nine': 9, \
                   'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, \
                   'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
                   'fifty': 50, \
                   'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
                   'million': 1000000}

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
        pred = " ".join(" = ".join(pred.split("=")).split())
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
                    if 0 <= number < 1:
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

def getMetaData():
    metadata = json.load(open("../../data/SCORE_json.json"))
    id2doi = {r['paper_id']: r['DOI_CR'] for r in metadata['data']}
    doi2id = {r['DOI_CR']: r['paper_id'] for r in metadata['data']}
    rppmap = json.load(open("../../data/doi_to_file_name_data.json"))
    rppid2doi = {r['file']: r['doi'] for r in rppmap}
    return id2doi, rppid2doi, doi2id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for flair")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint directory to load the model checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to store the model predictions")
    parser.add_argument("--gpu_device", type=str, default="cuda:0", help="GPU device number to use")
    parser.add_argument("--data_type", type=str, default="TA1", help="Options: RPP, TA1, biomed")
    parser.add_argument("--raw_data_path", type=str, default="../../data_processed/raw_all.txt", help="Raw data path to be tagged")
    parser.add_argument("--max_num_tokens", type=int, default=128, help="Maximum number of tokens to consider for each sentence")

    args = parser.parse_known_args()[0]
    print(args)

    flair.device = torch.device(args.gpu_device)
    id2doi, rppid2doi, doi2id = getMetaData()

    # Load the model:
    tagger = SequenceTagger.load(args.checkpoint_path)
    transformer_model_name = '-'.join(tagger.embeddings.name.split('-')[2:])
    print(transformer_model_name)

    # Reload Transformer Embeddings
    config = AutoConfig.from_pretrained(transformer_model_name, output_hidden_states=True)
    tagger.embeddings.model = AutoModel.from_pretrained(transformer_model_name, config=config)
    tagger.to(flair.device)

    data_source = args.data_type

    data, curr_data, curr_id = {}, [], None
    with open(args.raw_data_path, 'r') as f:
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

    # Sample (for testing)
    # new_data = {}
    # doi_val = "10.1016/j.obhdp.2010.10.001"
    # id_val = doi2id[doi_val]
    # new_data[id_val] = data[id_val]
    # data = new_data

    s_cnt = 1
    e_cnt = 10
    i_cnt = 1
    new_data = {}
    for k, v in data.items():
        if i_cnt >= s_cnt and i_cnt <=e_cnt:
            new_data[k] = v
        if i_cnt > e_cnt:
            break
        i_cnt += 1
    data = new_data

    all_pred = {}
    for cur_id, cur_sentences in tqdm(data.items()):
        curr_pred = defaultdict(lambda: [])
        claim2_sents_idx = []
        claim3a_sents_idx = []
        claim3b_sents_idx = []
        claim4_sents_idx = []
        for i in range(len(cur_sentences)):
            sent = cur_sentences[i]
            # for TA1 only: store the sentence indexes for the claims (2, 3a, 3b and 4)
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
            cleaned_sent = " ".join(nltk.word_tokenize(sent))
            cleaned_sentence = Sentence(cleaned_sent, use_tokenizer=False)

            # TODO: Check back later to see if we can trim/breakdown and not skip long sentences
            # Skip the sentences that are either empty or longer than the specified sequence:
            if len(cleaned_sentence.tokens) > args.max_num_tokens or len(cleaned_sentence.tokens) <= 1 or cleaned_sentence.tokens[-1].text not in ['.']:
                continue

            pred_output = tagger.predict(cleaned_sentence)
            pred_spans = cleaned_sentence.get_spans('ner')
            # for TA1 only: Store sentence indexes that contain the extracted features:
            for cur_span in pred_spans:
                curr_pred[cur_span.tag].append([cur_span.text, {"sent_idx": i}])

        # For TA1 only: calculate the feature distances to nearest claims
        curr_pred = dict(curr_pred)
        if data_source == 'TA1':
            for cur_tag in curr_pred.keys():
                for i in range(len(curr_pred[cur_tag])):

                    # Compute the distance to claim2:
                    if not claim2_sents_idx:
                        curr_pred[cur_tag][i][1]["claim2_distance"] = None # if claim is not found, set None
                    else:
                        sent_idx = curr_pred[cur_tag][i][1]["sent_idx"]
                        curr_pred[cur_tag][i][1]["claim2_distance"] = sorted([sent_idx - claim_idx for claim_idx in claim2_sents_idx], key=lambda x: abs(x))[0]

                    # Compute the distance to claim3a:
                    if not claim3a_sents_idx:
                        curr_pred[cur_tag][i][1]["claim3a_distance"] = None # if claim is not found, set None
                    else:
                        sent_idx = curr_pred[cur_tag][i][1]["sent_idx"]
                        curr_pred[cur_tag][i][1]["claim3a_distance"] = sorted([sent_idx - claim_idx for claim_idx in claim3a_sents_idx], key=lambda x: abs(x))[0]

                    # Compute the distance to claim3b:
                    if not claim3b_sents_idx:
                        curr_pred[cur_tag][i][1]["claim3b_distance"] = None # if claim is not found, set None
                    else:
                        sent_idx = curr_pred[cur_tag][i][1]["sent_idx"]
                        curr_pred[cur_tag][i][1]["claim3b_distance"] = sorted([sent_idx - claim_idx for claim_idx in claim3b_sents_idx], key=lambda x: abs(x))[0]

                    # Compute the distance to claim4:
                    if not claim4_sents_idx:
                        curr_pred[cur_tag][i][1]["claim4_distance"] = None # if claim is not found, set None
                    else:
                        sent_idx = curr_pred[cur_tag][i][1]["sent_idx"]
                        curr_pred[cur_tag][i][1]["claim4_distance"] = sorted([sent_idx - claim_idx for claim_idx in claim4_sents_idx], key=lambda x: abs(x))[0]

        all_pred[cur_id] = curr_pred

    output = {}
    for cur_id, cur_predictions in all_pred.items():
        curr_output = {}


        # Process the predictions for Sample Size (SS):
        if cur_predictions.get("SS") is not None:
            processed = [(process_pred('SS', r[0]), r[1]) for r in cur_predictions['SS']]
            processed = [r for r in processed if r[0] is not None]
            curr_output['Sample Sizes'] = processed
        else:
            curr_output['Sample Sizes'] = None


        # Process the predictions for Model Name (TN):
        if cur_predictions.get("TN") is not None:
            processed = [[process_pred('TN', r[0]), r[1]] for r in cur_predictions['TN']]
            processed = [r for r in processed if r[0] is not None]

            # Sort based on the length of the model name in descending order
            processed = sorted(processed, key=lambda x: len(x[0]), reverse=True)

            # Remove the duplicates. If a model name is a subset of another model name, remove that model name
            flag_dict = defaultdict(lambda: 0)
            for i in range(1, len(processed)):
                model_i_subset = set(re.findall('[^-^\s]+', processed[i][0]))
                for j in range(0, i):
                    if flag_dict[j] == 1:
                        continue

                    model_j_subset = set(re.findall('[^-^\s]+', processed[j][0]))
                    if model_i_subset.issubset(model_j_subset):
                        flag_dict[i] = 1
                        break

            for i in range(1, len(processed)):
                if flag_dict[i] == 1:
                    processed[i][0] = None

            processed = [(r[0], r[1]) for r in processed if r[0] is not None]
            curr_output['Model Names'] = processed
        else:
            curr_output['Model Names'] = None


        # Process the predictions for Models (TE):
        if cur_predictions.get("TE") is not None:
            processed = [process_pred('TE', r[0]) for r in cur_predictions['TE']]
            processed = [r for r in processed if r is not None]
            if processed:
                curr_output['Number of Models/Tests'] = max(processed)
            else:
                curr_output['Number of Models/Tests'] = None
        else:
            curr_output['Number of Models/Tests'] = None


        # Process the predictions for Studies (SD):
        if cur_predictions.get("SD") is not None:
            processed = [process_pred('SD', r[0]) for r in cur_predictions['SD']]
            processed = [r for r in processed if r is not None]
            if processed:
                curr_output['Number of Studies'] = max(processed)
            else:
                curr_output['Number of Studies'] = None
                curr_output['Number of Studies'] = None
        else:
            curr_output['Number of Studies'] = None

        # Process the predictions for P-values (PV):
        if cur_predictions.get("PV") is not None:
            processed = [[process_pred('PV', r[0]), r[1]] for r in cur_predictions['PV']]
            processed = [r for r in processed if r[0] is not None]

            try:
                processed = [(rr, r[1]) for r in processed for rr in r[0][1]]
            except:
                pass

            if len(processed) != 0:
                curr_output['P Values'] = processed
            else:
                curr_output['P Values'] = None
        else:
            curr_output['P Values'] = None

        # Process the predictions for Effect Size (ES):
        if cur_predictions.get("ES") is not None:
            processed = [[process_pred('ES', r[0]), r[1]] for r in cur_predictions['ES']]
            processed = [r for r in processed if r[0] is not None]

            try:
                processed = [[xx, x[1]] for x in processed for xx in x[0][1]]
            except:
                pass

            # r = [x for x in processed if x[0][0] == 'r']
            # R2 = [x for x in processed if x[0][0] == 'R2']
            # r = [[xx, x[1]] for x in r for xx in x[0][1]]
            # R2 = [[xx, x[1]] for x in R2 for xx in x[0][1]]

            if len(processed) != 0:
                curr_output['Effect Sizes'] = processed
            else:
                curr_output['Effect Sizes'] = None
        else:
            curr_output['Effect Sizes'] = None

        if data_source == 'TA1':
            cur_id = id2doi[cur_id]
        elif data_source == 'RPP':
            cur_id = rppid2doi[cur_id]

        output[cur_id] = curr_output

    # Write the output to the file
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)



