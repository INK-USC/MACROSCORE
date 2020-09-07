import numpy as np, json, random
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.neural_network import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.naive_bayes import *
import pandas as pd, os
from nltk.tokenize import word_tokenize

# paper embedding: glove + mean pool
word_emb_file = open("./data_processed/glove.6B.100d.txt").readlines()[1:]
word_emb_file = [r[:-1].split(" ") for r in word_emb_file]
word_list = [r[0] for r in word_emb_file]
word_emb = np.array([[float(rr) for rr in r[1:]] for r in word_emb_file])
word_list = ['<unk>'] + word_list
word_emb = np.vstack((word_emb.mean(0), word_emb))

word2idx = {k:v for v,k in enumerate(word_list)}
data = json.load(open('./data_processed/RPP_scienceparse_classify_data.parsed_rpp', 'r'))


def encode_paper(content):
    # encode all words as a whole
    encoding = np.zeros(word_emb.shape[1])
    for section in content:
        all_words = [r for rr in section['text'] for r in rr]
        for w in all_words:
            if w in word2idx:
                encoding += word2idx[w]
            else:
                encoding += word2idx['<unk>']
    
    encoding /= len(all_words)
    return encoding

psycho_features = pd.read_excel('./data/RPP_psycho_features.xlsx').to_dict()

psycho_tokens = {}
for k in psycho_features['Unnamed: 0'].keys():
    category = psycho_features['Unnamed: 0'][k]
    words = [psycho_features['Top-1 / e.g. 1'][k],psycho_features['Top-2 / e.g. 2'][k],psycho_features['Top-3 / e.g. 3'][k]
             ,psycho_features['Top-4 / e.g. 4'][k],psycho_features['Top-5 / e.g. 5'][k],psycho_features['Top-6 / e.g. 6'][k]
             ,psycho_features['Top-7 / e.g. 7'][k],psycho_features['Top-8 / e.g. 8'][k],psycho_features['Top-9 / e.g. 9'][k]
             ,psycho_features['Top-10 / e.g. 10'][k]]

    psycho_tokens[category] = words


def delete_tokens(content, tokens, token_category):
    all_tokens = list()
    for key, value in tokens.items():
        all_tokens.extend(value)
    all_tokens = set(all_tokens)

    for section in content:
        text_tokens = word_tokenize(section['text'])
        tokens_without_sw = [word for word in text_tokens if not word in all_tokens]
        recover_text = " ".join(tokens_without_sw)
        section['text'] = recover_text

    return content

def random_delete_tokens(content):
    all_tokens = list()
    for section in content:
        text_tokens = word_tokenize(section['text'])
        all_tokens.extend(text_tokens)
    all_tokens = set(all_tokens)
    percentage = int(len(all_tokens) * 0.5)
    sample_tokens = random.sample(all_tokens, percentage)
    print(len(all_tokens), percentage)

    for section in content:
        text_tokens = word_tokenize(section['text'])
        tokens_without_sw = [word for word in text_tokens if not word in sample_tokens]
        recover_text = " ".join(tokens_without_sw)
        section['text'] = recover_text

    return content


def str_(x):
    return '%.2f' % x

data_processed = [[encode_paper(random_delete_tokens(r['content'])), \
                   np.array([r['O_within_CI_R'], r['Meta_analysis_significant'], r['pvalue_label']]), \
                   r['Fold_Id']] for r in data]


# do 4-fold cv instead of default 10-fold cv
# result shows no difference between 4-fold and 10-fold
folds = list(range(len(data_processed)))
random.shuffle(folds)
fold_size = int(np.ceil(len(data_processed) / 4))
folds = [folds[(i*fold_size):((i+1)*fold_size)] for i in range(4)]
id2fold = {r:i for i in range(4) for r in folds[i]}
for i in range(len(data_processed)):
    data_processed[i][-1] = id2fold[i]


f = open('./data/RPP_classify_result_random_token.csv', 'w')
f.write(',Meta_analysis_significant,,,,,,O_within_CI_R,,,,,,pvalue_label,,,,,\n')
f.write('classifier,f1,std,precision,std,recall,std,f1,std,precision,std,recall,std,f1,std,precision,std,recall,std\n')
for name, classifier in [['MLPClassifier', MLPClassifier], ['DummyClassifier', DummyClassifier], \
                         ['KNeighborsClassifier', KNeighborsClassifier], \
                         ['RandomForestClassifier', RandomForestClassifier], \
                         ['SVC', SVC], ['GaussianNB', GaussianNB]]:
    result = {r:[] for r in ['acc', 'prec', 'recall', 'f1']}
    all_preds, all_labels = [], []
    for fold_id in range(4):
        test_data = [r for r in data_processed if r[2]==fold_id]
        train_data = [r for r in data_processed if r[2]!=fold_id]

        train_X = np.vstack([r[0] for r in train_data])
        test_X = np.vstack([r[0] for r in test_data])
        train_y = np.vstack([r[1] for r in train_data])
        test_y = np.vstack([r[1] for r in test_data])
        clf = MultiOutputClassifier(classifier())
        _ = clf.fit(train_X, train_y)
        preds = clf.predict(test_X)
        all_preds.append(preds)
        all_labels.append(test_y)
        result['acc'].append([accuracy_score([r[i] for r in test_y], [r[i] for r in preds]) for i in range(3)])
        result['prec'].append([precision_score([r[i] for r in test_y], [r[i] for r in preds]) for i in range(3)])
        result['recall'].append([recall_score([r[i] for r in test_y], [r[i] for r in preds]) for i in range(3)])
        result['f1'].append([f1_score([r[i] for r in test_y], [r[i] for r in preds]) for i in range(3)])
    
    result = {k:[[vv[i] for vv in v] for i in range(3)] for k,v in result.items()}
    result_agg = {k:[[np.mean(v[i]), np.std(v[i])] for i in range(3)] for k,v in result.items()}
    
    f.write(name+','+str_(result_agg['f1'][1][0])+','+str_(result_agg['f1'][1][1])+','+\
            str_(result_agg['prec'][1][0])+','+str_(result_agg['prec'][1][1])+','+\
            str_(result_agg['recall'][1][0])+','+str_(result_agg['recall'][1][1])+','+\
            str_(result_agg['f1'][0][0])+','+str_(result_agg['f1'][0][1])+','+\
            str_(result_agg['prec'][0][0])+','+str_(result_agg['prec'][0][1])+','+\
            str_(result_agg['recall'][0][0])+','+str_(result_agg['recall'][0][1])+','+\
            str_(result_agg['f1'][2][0])+','+str_(result_agg['f1'][2][1])+','+\
            str_(result_agg['prec'][2][0])+','+str_(result_agg['prec'][2][1])+','+\
            str_(result_agg['recall'][2][0])+','+str_(result_agg['recall'][2][1])+'\n')

f.close()

