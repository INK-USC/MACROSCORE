import numpy as np
from transformers import BertTokenizer, BertModel, LongformerTokenizer, LongformerModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random, math
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk

import os, sys
import argparse
np.random.seed(1337)

class REPRDataset(Dataset):
    def __init__(self, args, model_type, type_path="total"):
        # Getting the device info
        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Initialize the tokenizer and lm embeddings model
        if model_type == "M1" or model_type == "M2":
            self.lm_tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
            self.lm_embeddings_model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased").to(self.device)

        elif model_type == "M3":
            self.lm_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', do_lower_case=True)
            self.lm_embeddings_model = LongformerModel.from_pretrained('allenai/longformer-base-4096').to(self.device)

        # Initialize the feature type, window size
        if args.feature_type is not None:
            self.feature_type = args.feature_type
        else:
            if model_type == "M1":
                self.feature_type = "claim_only"
            elif model_type == "M2" or model_type == "M3":
                self.feature_type = "content"

        if args.window_size is not None:
            self.window_size = args.window_size
        else:
            if model_type == "M1" or model_type == "M2":
                self.window_size = 100
            elif model_type == "M3":
                self.window_size = 512

        self.inputs = []
        self.inputs_len = []
        self.targets = []
        self.foldids = []
        self.rawdata = []
        self.rawsections = []
        self.paperids = []
        self.args = args
        self.type_path = type_path
        self._build()

    def __getitem__(self, index):
        return {
            "fold_id": self.foldids[index],
            "inputs": self.inputs[index],
            "inputs_len": self.inputs_len[index],
            "targets": self.targets[index],
            "raw_inputs": self.rawdata[index],
            "raw_sections": self.rawsections[index],
            "paper_id": self.paperids[index]
        }

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == "total":
            data_path = os.path.join(self.args.data_dir, "data.json")
        elif self.type_path == "train":
            data_path = os.path.join(self.args.data_dir, "train.json")
        elif self.type_path == "dev":
            data_path = os.path.join(self.args.data_dir, "dev.json")
        else:
            raise ValueError("Type path argument is invalid")

        with open(data_path, "r") as f:
            data = json.load(f)

        training_data = []
        for cur_row in data:

            # Extract the input
            if self.feature_type == "content":
                content_val = cur_row['content']
            elif self.feature_type == "claim_only":
                if cur_row.get("claim2") is not None and cur_row.get("claim3a") is not None and cur_row.get(
                        "claim3b") is not None and cur_row.get("claim4") is not None:
                    claim2 = cur_row['claim2']
                    claim3a = cur_row['claim3a']
                    claim3b = cur_row['claim3b']
                    claim4 = cur_row['claim4']
                elif cur_row.get("coded_claim2") is not None and cur_row.get("coded_claim3a") is not None and cur_row.get(
                        "coded_claim3b") is not None and cur_row.get("coded_claim4") is not None:
                    claim2 = cur_row['coded_claim2']
                    claim3a = cur_row['coded_claim3a']
                    claim3b = cur_row['coded_claim3b']
                    claim4 = cur_row['coded_claim4']
                else:
                    print("Paper with id {} doesn't have claims. Skipping it".format(cur_row["paper_id"]))
                    continue
                content_val = [claim2, claim3a, claim3b, claim4]
            else:
                raise ValueError("Feature type argument is invalid")

            # Extract the output
            if self.args.data_type == "RPP":
                label_val = cur_row["Meta_analysis_significant"]
            elif self.args.data_type == "TA2":
                label_val = cur_row["label"]
            else:
                raise ValueError("Data type argument is invalid")

            # Extract the fold id
            fold_val = cur_row["Fold_Id"]

            # Extract the paper id
            if cur_row.get("DOI_CR") is not None:
                paper_id = cur_row["DOI_CR"]
            elif cur_row.get("doi") is not None:
                paper_id = cur_row["doi"]
            elif cur_row.get("paper_id") is not None:
                paper_id = cur_row["paper_id"]
            else:
                paper_id = None

            training_data.append([content_val, label_val, fold_val, paper_id])

        print("Loading the data from " + data_path)
        for idx in tqdm(range(len(training_data))):
            cur_input = training_data[idx][0]
            cur_target = training_data[idx][1]
            cur_fold = int(training_data[idx][2])
            cur_pid = training_data[idx][3]

            if self.args.loss_type == "classification":
                cur_target = 1 if cur_target >= 0.5 else 0
            self._create_features(cur_input, cur_target, cur_fold, cur_pid)

    def split_paper(self, content):
        l_total = []
        overlap_window_size = int(self.window_size/2)
        if len(content.split()) // overlap_window_size > 0:
            n = len(content.split()) // overlap_window_size
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_partial = content.split()[:self.window_size]
                l_total.append(" ".join(l_partial))
            else:
                l_partial = content.split()[w * overlap_window_size:w * overlap_window_size + self.window_size]
                l_total.append(" ".join(l_partial))
        return l_total

    def encode_paper(self, content):
        device = self.device
        embeddings = []
        for cur_section in content:
            split_sentences = self.split_paper(cur_section['text'])
            section_embeddings = []
            for sentence in split_sentences:
                input_ids = torch.tensor(self.lm_tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
                try:
                    input_ids = input_ids.to(device)
                    outputs = self.lm_embeddings_model(input_ids)
                except Exception as e:
                    print(e)
                    continue
                cls_embedding = outputs[0][:, 0, :].squeeze().detach().cpu()
                del outputs
                del input_ids
                section_embeddings.append(cls_embedding)
            section_final_embeddings = torch.mean(torch.stack(section_embeddings), 0)
            embeddings.append(section_final_embeddings)
        final_embs = torch.stack(embeddings)
        return final_embs, len(embeddings)

    def encode_paper_sentence_split(self, content):
        device = self.device
        embeddings = []
        raw_inputs = []
        raw_sections = []
        for section_idx, cur_section in enumerate(content):
            split_sentences = nltk.sent_tokenize(cur_section['text'])

            if cur_section.get("heading") is not None:
                cur_section_heading = cur_section["heading"]
            else:
                cur_section_heading = section_idx

            cur_sentence_tokens = []
            for sentence in split_sentences:
                cur_tokens = sentence.split()
                if len(cur_tokens) + len(cur_sentence_tokens) <= self.window_size:
                    cur_sentence_tokens += cur_tokens
                else:
                    if len(cur_sentence_tokens) != 0:
                        cur_sentence = " ".join(cur_sentence_tokens)
                        input_ids = torch.tensor(self.lm_tokenizer.encode(cur_sentence)).unsqueeze(0)  # Batch size 1
                        try:
                            input_ids = input_ids.to(device)
                            outputs = self.lm_embeddings_model(input_ids)
                        except Exception as e:
                            print(e)
                            continue
                        cls_embedding = outputs[0][:, 0, :].squeeze().detach().cpu()
                        del outputs
                        del input_ids
                        embeddings.append(cls_embedding)
                        raw_inputs.append(cur_sentence)
                        raw_sections.append(cur_section_heading)

                    # if a single sentence is more than window size (cut it down)
                    if len(cur_tokens) > self.window_size:
                        cur_sentence_tokens = cur_tokens[:self.window_size]
                    else:
                        cur_sentence_tokens = cur_tokens

        final_embs = torch.stack(embeddings)
        return final_embs, len(embeddings), raw_inputs, raw_sections

    def encode_paper_sentence_wise(self, content):
        device = self.device
        embeddings = []
        raw_inputs = []
        raw_sections = []
        for section_idx, cur_section in enumerate(content):
            if len(cur_section['text']) == 0:
                continue

            if cur_section.get("heading") is not None:
                cur_section_heading = cur_section["heading"]
            else:
                cur_section_heading = section_idx

            split_sentences = nltk.sent_tokenize(cur_section['text'])
            for sentence in split_sentences:
                cur_sentence_tokens = sentence.split()

                # if a single sentence is more than window size (cut it down)
                if len(cur_sentence_tokens) > self.window_size:
                    cur_sentence_tokens = cur_sentence_tokens[:self.window_size]
                else:
                    cur_sentence_tokens = cur_sentence_tokens

                # Filter out sentences with only urls/citations etc.
                if len(cur_sentence_tokens) >= 7:
                    cur_sentence = " ".join(cur_sentence_tokens)
                    input_ids = torch.tensor(self.lm_tokenizer.encode(cur_sentence)).unsqueeze(0)  # Batch size 1
                    try:
                        input_ids = input_ids.to(device)
                        outputs = self.lm_embeddings_model(input_ids)
                    except Exception as e:
                        print(e)
                        continue
                    cls_embedding = outputs[0][:, 0, :].squeeze().detach().cpu()
                    del outputs
                    del input_ids
                    embeddings.append(cls_embedding)
                    raw_inputs.append(cur_sentence)
                    raw_sections.append(cur_section_heading)

        final_embs = torch.stack(embeddings)
        return final_embs, len(embeddings), raw_inputs, raw_sections

    def encode_claims(self, claims):
        device = self.device
        embeddings = []
        raw_inputs = []
        for cur_claim in claims:
            input_ids = torch.tensor(self.lm_tokenizer.encode(cur_claim)).unsqueeze(0)  # Batch size 1
            try:
                input_ids = input_ids.to(device)
                outputs = self.lm_embeddings_model(input_ids)
            except Exception as e:
                print(e)
                continue
            cls_embeddings = outputs[0][:, 0, :].squeeze().detach().cpu()
            del outputs
            del input_ids
            embeddings.append(cls_embeddings)
            raw_inputs.append(cur_claim)
        final_embs = torch.stack(embeddings)
        return final_embs, len(embeddings), raw_inputs

    def _create_features(self, cur_input, cur_target, cur_fold, cur_pid):
        if self.feature_type == "content":
            final_embeddings, doc_len, raw_inputs, raw_sections = self.encode_paper_sentence_wise(cur_input)
        elif self.feature_type == "claim_only":
            final_embeddings, doc_len, raw_inputs = self.encode_claims(cur_input)
            raw_sections = None
        else:
            raise ValueError("Feature type argument is invalid")

        self.paperids.append(cur_pid)
        self.rawdata.append(raw_inputs)
        self.rawsections.append(raw_sections)
        self.inputs.append(final_embeddings)
        self.inputs_len.append(doc_len)
        self.targets.append(torch.tensor(cur_target))
        self.foldids.append(cur_fold)


class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embedding_length, bidirectional=False):
        super(AttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length

        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        if bidirectional:
            self.num_directions = 2
            self.out_hidden_size = hidden_size * 2
        else:
            self.num_directions = 1
            self.out_hidden_size = hidden_size

        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.label = nn.Linear(self.out_hidden_size, output_size)

    def attention_net(self, lstm_output, final_state):

        # hidden = (batch_size x out_hidden_size x 1)
        hidden = final_state.view(-1, self.out_hidden_size, 1)

        # attn_weights = (batch_size x seq_length)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)

        # soft_attn_weights = (batch_size x seq_length)
        soft_attn_weights = F.softmax(attn_weights, 1)

        # new_hidden_state = (batch_size x out_hidden_size)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state, soft_attn_weights

    def forward(self, input, input_lengths, batch_size=None):
        input.to(self.device)

        if batch_size is None:
            h_0 = torch.autograd.Variable(torch.zeros(self.num_directions, self.batch_size, self.hidden_size, device=self.device))
            c_0 = torch.autograd.Variable(torch.zeros(self.num_directions, self.batch_size, self.hidden_size, device=self.device))
        else:
            h_0 = torch.autograd.Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size, device=self.device))
            c_0 = torch.autograd.Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size, device=self.device))

        input_packed = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        output_packed, (final_hidden_state, final_cell_state) = self.lstm(input_packed, (h_0, c_0))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        output.to(self.device)
        attn_output, attn_weight = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits, attn_weight


def extractImportantSegment(model, dataset):
    result_list = []
    col_list = ["paper_id", "important_segment", "important_segment_idx", "important_section_heading_or_idx", "label", "Fold_Id"]
    for idx in tqdm(range(len(dataset)), desc="Imp Segment Extraction"):
        cur_pid = dataset[idx]["paper_id"]
        cur_fold_id = dataset[idx]["fold_id"]
        cur_inputs = dataset[idx]["inputs"].unsqueeze(0).to(device)
        cur_inputs_len = dataset[idx]["inputs_len"]
        cur_targets = dataset[idx]["targets"]
        cur_raw_inputs = dataset[idx]["raw_inputs"]
        cur_raw_sections = dataset[idx]["raw_sections"]

        with torch.no_grad():
            inputs_list = cur_inputs
            inputs_lengths = torch.FloatTensor([cur_inputs_len])
            inputs_padded_list = nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)
            prediction, attn_weight = model(inputs_padded_list, inputs_lengths, 1)


        values, indices = torch.max(attn_weight, 1)
        imp_seg_idx = indices.cpu().item()

        label = cur_targets.cpu().item()
        imp_seg = cur_raw_inputs[imp_seg_idx]
        imp_sec = cur_raw_sections[imp_seg_idx]
        result_list.append([cur_pid, imp_seg, imp_seg_idx, imp_sec, label, cur_fold_id])

    result_df = pd.DataFrame(data=result_list, columns=col_list)
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters:
    parser.add_argument("--model_type", type=str, default="M1",
                        help="Model type. M1: Scibert + claim only, M2: Scibert, M3: Longformer")
    parser.add_argument("--model_path", type=str, default="ta2_class_checkpoint/m2_folds_sentwise_v2/fold_3/best_model.tar",
                        help="Checkpoint path of the trained model")
    parser.add_argument("--loss_type", type=str, default="classification",
                        help="Loss type. Options: 1. regression and 2. classification")

    parser.add_argument("--data_type", type=str, default="TA2",
                        help="Data Type. Options: 1. RPP and 2. TA2")
    parser.add_argument("--data_dir", type=str, default="../data_processed/ta2_classify_folds",
                        help="Data directory which contains training data")
    parser.add_argument("--output_dir", type=str, default="scibert_ta2_claim_only",
                        help="Output directory to store the model checkpoints")

    # Optional parameters:
    parser.add_argument("--use_bilstm", action="store_true",
                        help="Whether to make the lstm bidirectional or not")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Window size to break the paper into segments. Set None to use the default settings based on model type")
    parser.add_argument("--feature_type", type=str, default=None,
                        help="Feature type. Options: 1. claim_only and 2. content to override the default settings of model type."
                             "Set None to use the default settings based on model type")


    args = parser.parse_known_args()[0]
    print(args)

    # Create the output directory if its not present
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    if args.loss_type == "regression":
        model = AttentionModel(1, 1, 100, 768, bidirectional=args.use_bilstm)
    elif args.loss_type == "classification":
        model = AttentionModel(1, 2, 100, 768, bidirectional=args.use_bilstm)
    else:
        raise ValueError("Loss type argument is invalid")

    # Load the checkpoint
    checkpoint_state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint_state["model"])
    model.to(device)

    # Load the dataset
    total_dataset = REPRDataset(args, model_type=args.model_type, type_path="total")

    # Extract the important segments
    result_df = extractImportantSegment(model, total_dataset)

    # Store the results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, "important_segments.json")
    result_df.to_json(output_path, orient="records")


