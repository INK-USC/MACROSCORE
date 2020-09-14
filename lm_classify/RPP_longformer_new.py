import numpy as np
np.random.seed(1337)
import numpy as np
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from sklearn.metrics import *
import os, gc

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')

save_dir = os.path.join("checkpoint", "save")
model_name = 'longformer_new'
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
model.to('cuda')

def encode_paper(content):
    sections = []
    word_length = 0
    for cont in content:
        word_length += len(cont['text'].split())
        if word_length > 2500:
            break
        sections.append(cont['text'])

    section_text = " ".join(sections)
    return section_text


def train_model(train_data, epoch):
    total_epoch_loss = 0
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    random.shuffle(train_data)
    for idx, data in enumerate(train_data):
        document = encode_paper(data[0])
        inputs = tokenizer(document, return_tensors="pt", truncation=True)
        labels = torch.LongTensor([data[1]]).unsqueeze(0)  # Batch size 1
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            labels = labels.cuda()

        optim.zero_grad()
        output = model(**inputs, labels=labels)
        loss, logits = output[:2]

        del inputs
        del labels
        del output

        loss.backward()
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

def eval_model(test_data):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_data):
            document = encode_paper(data[0])
            inputs = tokenizer(document, return_tensors="pt")
            labels = torch.LongTensor([data[1]]).unsqueeze(0)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                labels = labels.cuda()

            outputs = model(**inputs, labels=labels)
            loss, logits = outputs[:2]

            preds.append(torch.max(logits, 1)[1].view(labels.size()).data)
            targets.append(labels.data)

    return preds, targets

data = json.load(open('../data_processed/RPP_scienceparse_classify_data.json', 'r'))

meta_process = [[r['content'], r['Meta_analysis_significant'], r['Fold_Id']] for r in data]

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
        train_loss = train_model(train_data, epoch)
        print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}')

    preds, target = eval_model(test_data)
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
