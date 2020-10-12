import numpy as np
np.random.seed(1337)
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from sklearn.metrics import *
import os

# #Load the BERT tokenizer.
# print('Loading BERT tokenizer...')
# tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
# bertmodel = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
# bertmodel.to('cuda')
# save_dir = os.path.join("checkpoint", "save")
# model_name = 'scibert100_delpsycho'

def split_paper(content):
    l_total = []
    if len(content.split()) // 50 > 0:
        n = len(content.split()) // 50
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = content.split()[:100]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = content.split()[w * 50:w * 50 + 100]
            l_total.append(" ".join(l_parcial))

    return l_total

def encode_paper(content):
    embs = []
    for cont in content:
        split_sentences = split_paper(cont['text'])
        for sentence in split_sentences:
            input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0) # Batch size 1
            try:
                input_ids = input_ids.to('cuda')
                outputs = bertmodel(input_ids)
            except Exception as e:
                print(e)
                continue
            cls = outputs[0][:, 0, :].squeeze().detach().cpu()
            embs.append(cls)
    final_embs = torch.stack(embs)
    final_embs = final_embs.unsqueeze(0)
    return final_embs, len(embs)

class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embedding_length):
        super(AttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length

        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state, soft_attn_weights

    def forward(self, input, batch_size=None):
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            c_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output, attn_weight = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits, attn_weight

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_data, epoch):
    total_epoch_loss = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, data in enumerate(train_data):
        document = data[0]
        embs, doc_len = encode_paper(document)

        target = [data[1]]
        target = torch.LongTensor(target)
        if torch.cuda.is_available():
            embs = embs.cuda()
            target = target.cuda()

        optim.zero_grad()
        prediction, attn_weight = model(embs, 1)
        loss = F.cross_entropy(prediction, target)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        print (f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}')

        total_epoch_loss += loss.item()

    directory = os.path.join(save_dir, model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'iteration': epoch,
        'model': model.state_dict(),
        'opt': optim.state_dict(),
    }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))

    return total_epoch_loss

def eval_model(model, test_data):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_data):
            document = data[0]
            embs, doc_len = encode_paper(document)
            target = [data[1]]
            target = torch.LongTensor(target)
            if torch.cuda.is_available():
                embs = embs.cuda()
                target = target.cuda()

            prediction, attn_weight = model(embs,1)
            preds.append(torch.max(prediction, 1)[1].view(target.size()).data)
            targets.append(target.data)

    return preds, targets


import pandas as pd, os
from nltk.tokenize import word_tokenize
import random

psycho_features = pd.read_excel('../data/RPP_psycho_correlation.xlsx').to_dict()

psycho_tokens = {}
i = 0
for k in psycho_features['feature'].keys():
    if i == 50:
        break
    category = psycho_features['feature'][k]
    words = [psycho_features['example 1'][k],
             psycho_features['example 2'][k],
             psycho_features['example 3'][k],
             psycho_features['example 4'][k],
             psycho_features['example 5'][k],
             psycho_features['example 6'][k],
             psycho_features['example 7'][k],
             psycho_features['example 8'][k],
             psycho_features['example 9'][k],
             psycho_features['example 10'][k]]
    print(category)
    i+=1
    psycho_tokens[category] = words


def delete_tokens(content, tokens, token_category):
    all_tokens = list()
    for key, value in tokens.items():
        all_tokens.extend(value)
    all_tokens = set(all_tokens)
    del_total = 0
    total = 0
    for section in content:
        text_tokens = word_tokenize(section['text'])
        tokens_without_sw = [word for word in text_tokens if not word in all_tokens]
        del_num = len(text_tokens) - len(tokens_without_sw)
        del_total += del_num
        total += len(text_tokens)
        recover_text = " ".join(tokens_without_sw)
        section['text'] = recover_text

    return content, del_total, total

def random_delete_tokens(content, tokens, token_category):
    all_tokens = list()
    for key, value in tokens.items():
        all_tokens.extend(value)
    all_tokens = set(all_tokens)

    for section in content:
        text_tokens = word_tokenize(section['text'])

        tokens_without_sw = [word for word in text_tokens if not word in all_tokens]

        sorted_sample = [text_tokens[i] for i in sorted(random.sample(range(len(text_tokens)), len(tokens_without_sw)))]

        recover_text = " ".join(sorted_sample)
        section['text'] = recover_text

    return content

data = json.load(open('../data_processed/RPP_scienceparse_classify_data.json', 'r'))

del_t = 0
tot = 0
for r in data:
    content, del_total, total = delete_tokens(r['content'],psycho_tokens, None)
    del_t += del_total
    tot += total
print(del_t / tot, "faffa")


#r_process = [[r['content'], r['O_within_CI_R'], r['Fold_Id']] for r in data]
meta_process = [[random_delete_tokens(r['content'], psycho_tokens, None), r['Meta_analysis_significant'], r['Fold_Id']] for r in data]
#pvalue_process = [[r['content'], r['pvalue_label'], r['Fold_Id']] for r in data]


folds = list(range(len(meta_process)))
random.shuffle(folds)
fold_size = int(np.ceil(len(meta_process) / 4))
folds = [folds[(i*fold_size):((i+1)*fold_size)] for i in range(4)]
id2fold = {r:i for i in range(4) for r in folds[i]}

for i in range(len(meta_process)):
    meta_process[i][-1] = id2fold[i]

accuracy = []
precision = []
recall = []
f1 = []

for fold_id in range(4):
    model = AttentionModel(1, 2, 100, 768)
    test_data = [r for r in meta_process if r[2]==fold_id]
    train_data = [r for r in meta_process if r[2]!=fold_id]
    for epoch in range(10):
        train_loss = train_model(model, train_data, epoch)

        print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}')

    preds, target = eval_model(model, test_data)
    print("accuracy ", accuracy_score([r.cpu().item() for r in target], [r.cpu().item() for r in preds]))
    print("precision ", precision_score([r.cpu().item() for r in target], [r.cpu().item() for r in preds]))
    print("recall ", recall_score([r.cpu().item() for r in target], [r.cpu().item() for r in preds]))
    print("f1 ", f1_score([r.cpu().item() for r in target], [r.cpu().item() for r in preds]))

    accuracy.append(accuracy_score([r.cpu().item() for r in target], [r.cpu().item() for r in preds]))
    precision.append(precision_score([r.cpu().item() for r in target], [r.cpu().item() for r in preds]))
    recall.append(recall_score([r.cpu().item() for r in target], [r.cpu().item() for r in preds]))
    f1.append(f1_score([r.cpu().item() for r in target], [r.cpu().item() for r in preds]))

print("final_accuracy ", np.mean(accuracy))
print("final_precision ",  np.mean(precision))
print("final_recall ",  np.mean(recall))
print("final_f1 ",  np.mean(f1))
