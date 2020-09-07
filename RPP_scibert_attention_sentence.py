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

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
bertmodel = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
bertmodel.to('cuda')

save_dir = os.path.join("checkpoint", "save")
model_name = 'scibert100'

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


def encode_sentence(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0) # Batch size 1
    try:
        input_ids = input_ids.to('cuda')
        outputs = bertmodel(input_ids)
    except Exception as e:
        print(e)
    cls = outputs[0][:, 0, :].squeeze().detach().cpu()
    cls = cls.unsqueeze(0)
    return cls

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

def train_model_sentence(model, train_data, epoch):
    total_epoch_loss = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, data in enumerate(train_data):
        document = data[0]
        embs = encode_sentence(document)

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

    model_name = 'sentence'
    directory = os.path.join(save_dir, model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'iteration': epoch,
        'model': model.state_dict(),
        'opt': optim.state_dict(),
    }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))

    return total_epoch_loss


def eval_model_sentence(model, test_data):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_data):
            document = data[0]
            embs = encode_sentence(document)
            target = [data[1]]
            target = torch.LongTensor(target)
            if torch.cuda.is_available():
                embs = embs.cuda()
                target = target.cuda()

            prediction, attn_weight = model(embs,1)
            preds.append(torch.max(prediction, 1)[1].view(target.size()).data)
            targets.append(target.data)

    return preds, targets

def attention_analyze(model, test_data):
    model.eval()
    max_seg = []
    with torch.no_grad():
        for idx, data in enumerate(test_data):
            document = data[0]
            embs, doc_len = encode_paper(document)
            if torch.cuda.is_available():
                embs = embs.cuda()

            prediction, attn_weight = model(embs,1)
            values, indices = torch.max(attn_weight, 1)
            max_seg.append(indices.cpu().item())
    return max_seg

data = json.load(open('./data_processed/RPP_scienceparse_classify_data.parsed_rpp', 'r'))
print("data uploaded")
#r_process = [[r['content'], r['O_within_CI_R'], r['Fold_Id']] for r in data]
meta_process = [[r['content'], r['Meta_analysis_significant'], r['Fold_Id']] for r in data]
#pvalue_process = [[r['content'], r['pvalue_label'], r['Fold_Id']] for r in data]

#
loadFilename = os.path.join(save_dir, model_name, '7_checkpoint.tar')
if loadFilename:
    checkpoint = torch.load(loadFilename)
    model_checkpoint = checkpoint['model']
    opt_checkpoint = checkpoint['opt']

model = AttentionModel(1, 2, 100, 768)
if loadFilename:
    model.load_state_dict(model_checkpoint)
    model.cuda()

max_seg = attention_analyze(model, meta_process)
print(max_seg)

# max_seg = [30, 27, 82, 12, 15, 20, 63, 11, 28, 31, 29, 31,
# 25, 20, 91, 37, 11, 27, 50, 30, 17, 25, 116, 82, 141, 59, 48,
# 115, 89, 59, 55, 90, 8, 84, 25, 133, 162, 57, 79, 29, 12, 10,
# 11, 5, 6, 8, 7, 5, 25, 11, 19, 10, 13, 16, 18, 7, 14, 15, 14, 13, 11, 19, 20, 24, 10, 10, 8]
#
# ## max_seg = [87, 107, 87, 12, 14, 20, 63, 19, 72, 31, 47, 31, 24, 72,
# # 91, 37, 13, 46, 50, 28, 56, 42, 116, 76, 141, 53, 47, 116, 89, 59,
# # 54, 90, 12, 84, 26, 136, 162, 56, 79, 29, 21, 28, 26, 10, 7, 8, 14,
# # 5, 25, 7, 8, 10, 10, 16, 7, 30, 15, 13, 29, 12, 11, 19, 22, 24, 10, 10, 8]
# #

meta_process = []
for r, i in zip(data, max_seg):
    splitted = []
    for cont in r['content']:
        splits = split_paper(cont['text'])
        splitted.extend(splits)

    meta_process.append([splitted[i], r['Meta_analysis_significant'], r['Fold_Id']])
    print(splitted[i])
    print(r['Meta_analysis_significant'])
#
#
class SimpleClassification(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassification, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x, batch_size):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out, batch_size

model = SimpleClassification(768, 100, 2)

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
    test_data = [r for r in meta_process if r[2]==fold_id]
    train_data = [r for r in meta_process if r[2]!=fold_id]
    for epoch in range(10):
        train_loss = train_model_sentence(model, train_data, epoch)

        print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}')

    preds, target = eval_model_sentence(model, test_data)
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
