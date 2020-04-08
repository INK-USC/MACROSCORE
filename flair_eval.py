from flair.data import Sentence
from flair.models import SequenceTagger
import argparse

def iob_to_chunk(labels):
    all_chunks, curr_chunk = [], ()
    in_ent = False
    for i in range(len(labels)+1):
        if i == len(labels):
            if in_ent:
                curr_chunk += (i-1,)
                all_chunks.append(curr_chunk)
                curr_chunk = ()
                in_ent = False
            break
        if labels[i][0] == "I":
            continue
        elif labels[i][0] == "O":
            if in_ent:
                curr_chunk += (i-1,)
                all_chunks.append(curr_chunk)
                curr_chunk = ()
                in_ent = False
        elif labels[i][0] == "B":
            if in_ent:
                curr_chunk += (i-1,)
                all_chunks.append(curr_chunk)
                curr_chunk = ()
                in_ent = False
            curr_chunk = (labels[i].split('-')[1], i)
            in_ent = True
    return all_chunks

def word_overlap(true_spans, pred_spans):
    if not true_spans and not pred_spans:
        return 0, 0, 0
    
    max_len = max([r[2] for r in true_spans+pred_spans]) + 1
    pseudo_true_labels = ['O' for r in range(max_len)]
    pseudo_pred_labels = ['O' for r in range(max_len)]
    for s in true_spans:
        for i in range(s[1],s[2]+1):
            pseudo_true_labels[i] = s[0]
    
    for s in pred_spans:
        for i in range(s[1],s[2]+1):
            pseudo_pred_labels[i] = s[0]
    
    TP, FP, FN = 0, 0, 0
    for i in range(len(pseudo_true_labels)):
        if pseudo_true_labels[i] == pseudo_pred_labels[i] and pseudo_pred_labels[i] != 'O':
            TP += 1
        elif pseudo_true_labels[i] != pseudo_pred_labels[i] and pseudo_pred_labels[i] == 'O':
            FN += 1
        elif pseudo_true_labels[i] != pseudo_pred_labels[i] and pseudo_pred_labels[i] != 'O':
            FP += 1
        else:
            pass
    
    return TP, FP, FN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train flair")
    parser.add_argument("--folder", type=str, help="folder to chkp")
    parser.add_argument("--method", choices=['chunk', 'word'])
    args = parser.parse_args()
    args = vars(args)
    
    tagger = SequenceTagger.load('./flair_models/'+args['folder']+'/final-model.pt')
    test_data_f = "score/eng.testb.src"
    test_labels_f = "score/eng.testb.trg"
    TP, FP, FN = 0, 0, 0
    with open(test_data_f, 'r') as f_data,\
         open(test_labels_f, 'r') as f_labels:
        for sent, labels in zip(f_data, f_labels):
            sent = sent[:-1].split(' ')
            labels = labels[:-1].split(' ')
            true_spans = iob_to_chunk(labels)
            sentence = Sentence(' '.join(sent))
            tagger.predict(sentence)
            pred_spans = sentence.get_spans('ner')
            pred_spans = [(r.tag, r.tokens[0].idx-1, r.tokens[-1].idx-1) for r in pred_spans]
            if args['method'] == 'chunk':
                curr_TP = len(set(true_spans).intersection(pred_spans))
                curr_FP = len(pred_spans) - curr_TP
                curr_FN = len(true_spans) - curr_TP
            else:
                curr_TP, curr_FP, curr_FN = word_overlap(true_spans, pred_spans)
            
            TP += curr_TP
            FP += curr_FP
            FN += curr_FN

    P = TP / (TP+FP) if (TP+FP) else 0
    R = TP / (TP+FN) if (TP+FN) else 0
    F1 = 2 * P * R / (P+R) if (P+R) else 0
    print("P:", P, "\nR:", R, "\nF:", F1)
    print()
    
    all_labels = [r[:-1].split(' ') for r in open(test_labels_f, 'r').readlines()]
    all_labels = list(set([r for rr in all_labels for r in rr]))
    all_types = set([r[2:] for r in all_labels if r != 'O'])
    for type in sorted(all_types):
        num_true, num_pred = 0, 0
        TP, FP, FN = 0, 0, 0
        with open(test_data_f, 'r') as f_data,\
             open(test_labels_f, 'r') as f_labels:
            for sent, labels in zip(f_data, f_labels):
                sent = sent[:-1].split(' ')
                labels = labels[:-1].split(' ')
                true_spans = iob_to_chunk(labels)
                sentence = Sentence(' '.join(sent))
                tagger.predict(sentence)
                pred_spans = sentence.get_spans('ner')
                pred_spans = [(r.tag, r.tokens[0].idx-1, r.tokens[-1].idx-1) for r in pred_spans]
                true_spans = [r for r in true_spans if r[0] == type]
                pred_spans = [r for r in pred_spans if r[0] == type]
                if args['method'] == 'chunk':
                    curr_TP = len(set(true_spans).intersection(pred_spans))
                    curr_FP = len(pred_spans) - curr_TP
                    curr_FN = len(true_spans) - curr_TP
                else:
                    curr_TP, curr_FP, curr_FN = word_overlap(true_spans, pred_spans)
                
                TP += curr_TP
                FP += curr_FP
                FN += curr_FN
                
                num_true += len(true_spans)
                num_pred += len(pred_spans)
        
        P = TP / (TP+FP) if (TP+FP) else 0
        R = TP / (TP+FN) if (TP+FN) else 0
        F1 = 2 * P * R / (P+R) if (P+R) else 0
        print(type, "\nP:", P, "\nR:", R, "\nF:", F1)
        print("Num True:", num_true, "\nNum Pred:", num_pred)
        print()
        


