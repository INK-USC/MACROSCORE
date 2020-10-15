import numpy as np
np.random.seed(1337)
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os
import pandas as pd
from tqdm import tqdm

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

def encode_paper(content, bertmodel, tokenizer):
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    embs = []
    for cont in content:
        split_sentences = split_paper(cont['text'])
        section_embs = []
        for sentence in split_sentences:
            input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
            try:
                input_ids = input_ids.to(device)
                outputs = bertmodel(input_ids)
            except Exception as e:
                print(e)
                continue
            cls = outputs[0][:, 0, :].squeeze().detach().cpu()
            del outputs
            del input_ids
            section_embs.append(cls)
        section_final_embs = torch.mean(torch.stack(section_embs), 0)
        embs.append(section_final_embs)
    final_embs = torch.stack(embs)
    final_embs = final_embs.unsqueeze(0)
    return final_embs, len(embs)

def encode_sentence(sentence, bertmodel, tokenizer):
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0) # Batch size 1
    try:
        input_ids = input_ids.to(device)
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

        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def attention_net(self, lstm_output, final_state):

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state, soft_attn_weights

    def forward(self, input, batch_size=None):
        input = input.permute(1, 0, 2)
        input.to(self.device)

        if batch_size is None:
            h_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))
            c_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))
        else:
            h_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size, device=self.device))
            c_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size, device=self.device))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        output.to(self.device)
        attn_output, attn_weight = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits, attn_weight

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model_sentence(model, train_data, epoch, bertmodel, tokenizer, args, show_every=50):
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    total_epoch_loss = 0
    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, data in enumerate(train_data):
        document = data[0]
        embs = encode_sentence(document, bertmodel, tokenizer)

        target = [data[1]]
        target = torch.LongTensor(target)

        if device == "cuda":
            embs = embs.cuda()
            target = target.cuda()

        optim.zero_grad()
        prediction, attn_weight = model(embs, 1)
        loss = F.cross_entropy(prediction, target)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if idx % show_every == 0:
            print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}')
        total_epoch_loss += loss.item()

    directory = args.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save({
        'iteration': epoch,
        'model': model.state_dict(),
        'opt': optim.state_dict(),
    }, os.path.join(directory, 'model_ckpt_epoch={}.tar'.format(epoch)))

    return total_epoch_loss


def eval_model_sentence(model, test_data, bertmodel, tokenizer, args):
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_data):
            document = data[0]
            embs = encode_sentence(document, bertmodel, tokenizer)
            target = [data[1]]
            target = torch.LongTensor(target)

            if device == "cuda":
                embs = embs.cuda()
                target = target.cuda()

            prediction, attn_weight = model(embs,1)
            preds.append(torch.max(prediction, 1)[1].view(target.size()).data)
            targets.append(target.data)

    return preds, targets

def attention_analyze(model, test_data, bertmodel, tokenizer):
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    print("Inside attention analyse!!!")
    print("Total samples = ", len(test_data))
    model.eval()
    max_seg = []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_data)):
            document = data[0]
            embs, doc_len = encode_paper(document, bertmodel, tokenizer)

            if device == "cuda":
                embs = embs.cuda()

            prediction, attn_weight = model(embs,1)
            values, indices = torch.max(attn_weight, 1)
            max_seg.append(indices.cpu().item())
    return max_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, default="allenai/scibert_scivocab_uncased",
                        help="Tokenizer name or path")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased",
                        help="Model name or path")
    parser.add_argument("--data_type", type=str, default="RPP",
                        help="Data Type. Options: 1. RPP and 2. TA2")
    parser.add_argument("--feature_type", type=str, default="content",
                        help="Feature type. Options: 1. claim_only and 2. content")
    parser.add_argument("--data_path", type=str, default="../data_processed/RPP_scienceparse_classify_data.json",
                        help="Data path which contains training data")
    parser.add_argument("--checkpoint_path", type=str, default="../scibert.tar",
                        help="Checkpoint path which contains the trained model")
    parser.add_argument("--output_path", type=str, default="../data_processed/RPP_scienceparse_classify_data_with_imp_segments_only.json",
                        help="Output path to write the output.")
    args = parser.parse_known_args()[0]
    print(args)

    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=True)
    bertmodel = BertModel.from_pretrained(args.model_name)
    bertmodel.to(device)

    with open(args.data_path, "r") as f:
        data = json.load(f)

    meta_process = []
    for r in data:

        if args.feature_type == "content":
            content_val = r['content']
        elif args.feature_type == "claim_only":
            if r.get("claim2") is not None:
                content_val = [r['claim2'], r['claim3a'], r['claim3b'], r['claim4']]
            elif r.get("coded_claim2") is not None:
                content_val = [r['coded_claim2'], r['coded_claim3a'], r['coded_claim3b'], r['coded_claim4']]
            else:
                content_val = None
                print("Claims are not present!")
        else:
            raise ValueError("Feature type argument is invalid")

        if args.data_type == "RPP":
            label_val = r["Meta_analysis_significant"]
            fold_id_val = r["Fold_Id"]
        elif args.data_type == "TA2":
            label_val = r["label"]
            fold_id_val = r["Fold_Id"]
        else:
            raise ValueError("Data type argument is invalid")

        meta_process.append([content_val, label_val, fold_id_val])

    model = AttentionModel(1, 2, 100, 768)
    loadFilename = args.checkpoint_path
    if loadFilename:
        checkpoint = torch.load(loadFilename, map_location=device)
        model_checkpoint = checkpoint['model']
        opt_checkpoint = checkpoint['opt']
        model.load_state_dict(model_checkpoint)
        model.to(device)

    max_seg = attention_analyze(model, meta_process, bertmodel, tokenizer)
    print(max_seg[0:5], len(max_seg))

    meta_process = []
    for r, i in zip(data, max_seg):
        splitted = []
        for cont in r['content']:
            splits = split_paper(cont['text'])
            splitted.extend(splits)

        meta_process.append([r['paper_id'], splitted[i], r['Meta_analysis_significant'], r['Fold_Id']])


    df = pd.DataFrame(meta_process, columns=["paper_id", "important_segment", "label", "Fold_Id"])
    df.to_csv(args.output_path, index=False)

