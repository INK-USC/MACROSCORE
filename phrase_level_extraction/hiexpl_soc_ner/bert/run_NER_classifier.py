from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, accuracy_score, f1_score
import torch
import argparse
import os, json
import datetime, time
from tqdm import tqdm
import copy
import torch.nn.functional as F

class BertForNER(BertForTokenClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None):
        if torch.cuda.device_count() > 0:
            device = "cuda"
        else:
            device = "cpu"

        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

class NERDataset(Dataset):
    def __init__(self, args, tokenizer, type_path="test"):
        self.input_ids = []
        self.input_mask = []
        self.segment_ids = []
        self.valid_positions = []
        self.label_ids = []
        self.label_mask = []

        self.label2idx = {}
        self.idx2label = {}

        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path

        self._create_label_map()
        self._build()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.input_ids[index]),
            "input_mask":  torch.tensor(self.input_mask[index]),
            "segment_ids":  torch.tensor(self.segment_ids[index]),
            "valid_positions":  torch.tensor(self.valid_positions[index]),
            "label_ids":  torch.tensor(self.label_ids[index]),
            "label_mask":  torch.tensor(self.label_mask[index]),
        }

    def _tokenize_sentence(self, cur_sentence_seq):
        cur_tokenized_sent_seq = []
        valid_positions = []
        for cur_word in cur_sentence_seq:
            tokenized_word = self.tokenizer.tokenize(cur_word)
            n_subwords = len(tokenized_word)

            for i in range(n_subwords):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)

            cur_tokenized_sent_seq += tokenized_word
        assert (len(cur_tokenized_sent_seq) == len(valid_positions))
        return cur_tokenized_sent_seq, valid_positions

    def _pre_process_input(self, cur_sentence_seq, cur_label_seq):
        tokens, valid_positions = self._tokenize_sentence(cur_sentence_seq)
        label_ids = cur_label_seq

        if len(tokens) >= self.args.max_seq_length - 1:
            tokens = tokens[0:(self.args.max_seq_length - 2)]
            label_ids = label_ids[0:(self.args.max_seq_length - 2)]
            valid_positions = valid_positions[0:(self.args.max_seq_length - 2)]

        # Insert "[CLS]"
        tokens.insert(0, "[CLS]")
        valid_positions.insert(0, 1)
        label_ids.insert(0, self.label2idx["[CLS]"])

        # Insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        label_ids.append(self.label2idx["[SEP]"])

        # Get ids
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)

        # Pad till the max_seq_length
        while len(input_ids) < self.args.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(1)
            label_ids.append(0)
            label_mask.append(0)

        while len(label_ids) < self.args.max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length
        assert len(valid_positions) == self.args.max_seq_length
        assert len(label_ids) == self.args.max_seq_length
        assert len(label_mask) == self.args.max_seq_length

        return input_ids, input_mask, segment_ids, valid_positions, label_ids, label_mask

    def _create_label_map(self):
        # Create the label map
        special_label_set = ['[PAD]', 'O', '[CLS]', '[SEP]']
        data_path = os.path.join(self.args.data_dir, "train.txt")
        label_col_index = -1

        with open(data_path, "r") as f:
            data_lines = f.readlines()

        label_set = set()
        for cur_line in data_lines:
            cur_line = cur_line.strip()
            if len(cur_line) == 0 or cur_line.startswith('-DOCSTART'):
                # End of a sentence
                continue
            else:
                try:
                    cur_label_split = cur_line.split()[label_col_index].split("-")
                    if len(cur_label_split) == 2:
                        label_set.add("-".join(cur_label_split))
                except Exception as e:
                    pass

        label_tags = list(sorted(set([x.split("-")[-1] for x in label_set])))
        label_set = []
        for tag in label_tags:
            label_set.append("B-" + tag)
            label_set.append("I-" + tag)
        label_set = special_label_set[:2] + label_set + special_label_set[2:]

        for idx, val in enumerate(label_set):
            self.label2idx[val] = idx
            self.idx2label[idx] = val

    def get_label_map(self):
        return self.label2idx, self.idx2label

    def _build(self):
        if self.type_path == "train":
            data_path = os.path.join(self.args.data_dir, "train.txt")
        else:
            data_path = os.path.join(self.args.data_dir, "test.txt")

        with open(data_path, "r") as f:
            data_lines = f.readlines()

        # Used for debugging (Set -1 to read the entire dataset)
        num_sentences_to_read = -1
        label_col_index = -1

        sentence_data_all = []
        cur_sentence_seq = []
        cur_label_seq = []
        cur_sent_num = 0
        for cur_line in data_lines:
            cur_line = cur_line.strip()
            if len(cur_line) == 0 or cur_line.startswith('-DOCSTART'):
                # End of a sentence
                if len(cur_sentence_seq) != 0:
                    sentence_data_all.append([cur_sentence_seq, cur_label_seq])

                    cur_sent_num += 1
                    if num_sentences_to_read != -1 and cur_sent_num >= num_sentences_to_read:
                        break

                # Reset the sequences
                cur_sentence_seq = []
                cur_label_seq = []
            else:

                try:
                    cur_input = cur_line.split()[0]
                    cur_label = cur_line.split()[label_col_index]
                    cur_sentence_seq.append(cur_input)
                    cur_label_seq.append(self.label2idx[cur_label])
                except Exception as e:
                    pass

        # Preprocess the data
        for idx, cur_row in enumerate(tqdm(sentence_data_all, desc="Load data")):
            cur_sentence_seq = cur_row[0]
            cur_label_seq = cur_row[1]
            input_ids, input_mask, segment_ids, valid_positions, label_ids, label_mask = self._pre_process_input(
                cur_sentence_seq, cur_label_seq)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.segment_ids.append(segment_ids)
            self.valid_positions.append(valid_positions)
            self.label_ids.append(label_ids)
            self.label_mask.append(label_mask)


def saveModel(model, model_suffix, args):
    print("Saving model = ", "model" + str(model_suffix))

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save the model:
    model.save_pretrained(os.path.join(args.output_dir, "model" + str(model_suffix)))

    # Save the model configuration:
    with open(os.path.join(args.output_dir, "model" + str(model_suffix) + "_params.json"), "w") as f:
        json.dump(vars(args), f)

def format_time(elapsed_seconds):
    elapsed_rounded = int(round((elapsed_seconds)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def computeMetrics(y_pred, y_true):
    try:
        accuracy = accuracy_score(y_true, y_pred)
    except Exception as e:
        print(e)
        accuracy = -1

    try:
        f1_val = f1_score(y_true, y_pred)
    except Exception as e:
        print(e)
        f1_val = -1

    print(classification_report(y_true, y_pred))
    return accuracy, f1_val


def train(model, train_dl, val_dl, args, idx2label_map):
    # Get the device info:
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    # Create the optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    total_training_steps = len(train_dl) * args.num_epochs
    warmup_steps = int(total_training_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_training_steps)

    training_logs = []
    inital_time = time.time()
    minimum_val_loss = None
    best_model_state_dict = None
    best_model_epoch = None
    model.to(device)
    for cur_epoch in tqdm(range(args.num_epochs)):
        print("\nEpoch number = ", cur_epoch, "\n")

        print("\nTraining...")
        start_time = time.time()
        total_train_loss = 0

        model.train()
        for cur_step, cur_batch in enumerate(tqdm(train_dl)):
            input_ids = cur_batch["input_ids"].to(device)
            input_mask = cur_batch["input_mask"].to(device)
            segment_ids = cur_batch["segment_ids"].to(device)
            valid_positions = cur_batch["valid_positions"].to(device)
            label_ids = cur_batch["label_ids"].to(device)
            label_mask = cur_batch["label_mask"].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids=input_ids,
                                 token_type_ids=segment_ids,
                                 attention_mask=input_mask,
                                 labels=label_ids,
                                 valid_ids=valid_positions,
                                 attention_mask_label=label_mask)
            total_train_loss += loss.item()

            # Backprop the loss
            loss.backward()

            # Perform Gradient Clipping to 1.0:
            clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters:
            optimizer.step()

            # Update learning rate:
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dl)
        total_train_time = format_time(time.time() - start_time)
        print("Training time for Epoch ", cur_epoch, " = ", total_train_time)
        print("Average training loss = ", avg_train_loss)

        print("\n\nValidation...")
        start_time = time.time()
        total_val_loss = 0
        y_true = []
        y_pred = []

        # Get [SEP] label index:
        sep_label_idx = None
        for k, v in idx2label_map.items():
            if v == "[SEP]":
                sep_label_idx = k

        # Setting the model in eval mode:
        model.eval()
        for cur_step, cur_batch in enumerate(tqdm(val_dl)):
            input_ids = cur_batch["input_ids"].to(device)
            input_mask = cur_batch["input_mask"].to(device)
            segment_ids = cur_batch["segment_ids"].to(device)
            valid_positions = cur_batch["valid_positions"].to(device)
            label_ids = cur_batch["label_ids"].to(device)
            label_mask = cur_batch["label_mask"].to(device)

            with torch.no_grad():
                loss, logits = model(input_ids=input_ids,
                                     token_type_ids=segment_ids,
                                     attention_mask=input_mask,
                                     labels=label_ids,
                                     valid_ids=valid_positions,
                                     attention_mask_label=label_mask)

                total_val_loss += loss.item()

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == sep_label_idx: # Reached the [SEP] label
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            temp_1.append(idx2label_map[label_ids[i][j]])
                            temp_2.append(idx2label_map[logits[i][j]])


        avg_val_loss = total_val_loss / len(val_dl)
        val_accuracy, val_f1_score = computeMetrics(y_pred, y_true)
        total_val_time = format_time(time.time() - start_time)
        print("Validation time for Epoch ", cur_epoch, " = ", total_val_time)
        print("Average validation loss = ", avg_val_loss)
        print("Validation accuracy = ", val_accuracy)
        print("Validation F1 score (micro) = ", val_f1_score)

        # Saving the model with minimum validation loss (best model till the current epoch)
        if minimum_val_loss is None:
            minimum_val_loss = avg_val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_model_epoch = cur_epoch
            saveModel(model, "", args)
        elif avg_val_loss < minimum_val_loss:
            minimum_val_loss = avg_val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_model_epoch = cur_epoch
            saveModel(model, "", args)

        training_logs.append(
            {
                "epoch": cur_epoch,
                "avg_train_loss": avg_train_loss,
                "train_time": total_train_time,
                "avg_val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1_score,
                "val_time": total_val_time,
            })

    print("Training done...")
    total_time = format_time(time.time() - inital_time)
    print("Total time taken = ", total_time)

    # Save the training logs:
    with open(os.path.join(args.output_dir, "training_logs.json"), "w") as f:
        json.dump(training_logs, f)

    # Save the model:
    print("Best model epoch = ", best_model_epoch)
    model.load_state_dict(best_model_state_dict)
    saveModel(model, "_best", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/ner_dataset",
                        help='Path for Data files')
    parser.add_argument('--output_dir', type=str, default="../models/repr_bert_ner_seq",
                        help='Path to save the checkpoints')
    parser.add_argument('--model_name_or_path', type=str, default="allenai/scibert_scivocab_uncased",
                        help='Model name or Path')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="allenai/scibert_scivocab_uncased",
                        help='Tokenizer name or Path')

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    args = parser.parse_known_args()[0]
    print(args)

    bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)

    # Create the data loaders
    train_dataset = NERDataset(args, bert_tokenizer, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    test_dataset = NERDataset(args, bert_tokenizer, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    print("Number of Train instances = ", len(train_dataset))
    print("Number of Test instances = ", len(test_dataset))

    # Get the label map
    label2idx_map, idx2label_map = train_dataset.get_label_map()
    num_labels = len(label2idx_map.keys())
    print("Number of labels = ", num_labels)
    print("Label map: \n", label2idx_map)

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task="ner")
    model = BertForNER.from_pretrained(args.model_name_or_path, config=config)
    train(model, train_dataloader, test_dataloader, args, idx2label_map)






