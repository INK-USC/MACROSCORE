import numpy as np
from transformers import BertTokenizer, BertModel, LongformerTokenizer, LongformerModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random, math
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader
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
        self.targets = []
        self.foldids = []
        self.paperids = []
        self.args = args
        self.type_path = type_path
        self._build()

    def __getitem__(self, index):
        return {
            "fold_id": self.foldids[index],
            "inputs": self.inputs[index],
            "targets": self.targets[index],
            "paper_ids": self.paperids[index]
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
        for cur_section in content:
            split_sentences = nltk.sent_tokenize(cur_section['text'])
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

                    # if a single sentence is more than window size (cut it down)
                    if len(cur_tokens) > self.window_size:
                        cur_sentence_tokens = cur_tokens[:self.window_size]
                    else:
                        cur_sentence_tokens = cur_tokens

        final_embs = torch.stack(embeddings)
        return final_embs, len(embeddings)

    def encode_claims(self, claims):
        device = self.device
        embeddings = []
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
        final_embs = torch.stack(embeddings)
        return final_embs, len(embeddings)

    def _create_features(self, cur_input, cur_target, cur_fold, cur_pid):
        if self.feature_type == "content":
            final_embeddings, doc_len = self.encode_paper_sentence_split(cur_input)
        elif self.feature_type == "claim_only":
            final_embeddings, doc_len = self.encode_claims(cur_input)
        else:
            raise ValueError("Feature type argument is invalid")

        self.inputs.append(final_embeddings)
        self.targets.append(torch.tensor(cur_target))
        self.foldids.append(cur_fold)
        self.paperids.append(cur_pid)


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

        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.label = nn.Linear(hidden_size, output_size)

    def attention_net(self, lstm_output, final_state):

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state, soft_attn_weights

    def forward(self, input, input_lengths, batch_size=None):
        input.to(self.device)

        if batch_size is None:
            h_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))
            c_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))
        else:
            h_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size, device=self.device))
            c_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size, device=self.device))

        input_packed = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        output_packed, (final_hidden_state, final_cell_state) = self.lstm(input_packed, (h_0, c_0))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        output.to(self.device)
        attn_output, attn_weight = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits, attn_weight

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def computeMetrics(y_true, y_pred, loss_type):
    metrics_dict = {}
    if loss_type == "regression":
        metrics_dict["rmse"] = math.sqrt(mean_squared_error(y_true, y_pred))
    elif loss_type == "classification":
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)
        metrics_dict["precision"] = precision_score(y_true, y_pred)
        metrics_dict["recall"] = recall_score(y_true, y_pred)
        metrics_dict["f1"] = f1_score(y_true, y_pred)
    else:
        metrics_dict = None

    return metrics_dict

def train(model, train_data_loader, val_data_loader, args, fold_id):
    training_logs = []
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    best_epoch = None
    best_val_loss = None
    best_metrics = None
    best_predictions = None
    paper_ids_list = None

    # Create the fold output directory
    if fold_id != -1:
        output_dir = os.path.join(args.output_dir, "fold_" + str(fold_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    model.to(device)
    if args.loss_type == "regression":
        loss_criterion = nn.MSELoss()
    elif args.loss_type == "classification":
        loss_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Loss type argument is invalid")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    for epoch in range(args.num_epochs):

        # Train
        model.train()
        avg_training_loss = 0.0
        num_train_steps = 1
        for batch_idx, cur_batch in enumerate(tqdm(train_data_loader, desc="Train Epoch {}".format(epoch))):
            inputs = cur_batch["inputs"].to(device)
            inputs_lengths = cur_batch["inputs_lengths"].to(device)
            targets = cur_batch["targets"].to(device)

            if args.loss_type == "regression":
                targets = targets.view((-1, 1))

            if args.loss_type == "classification":
                targets = targets.type(torch.LongTensor).to(device)

            predictions, attn_weight = model(inputs, inputs_lengths, batch_size=inputs.shape[0])

            loss = loss_criterion(predictions, targets)
            avg_training_loss += loss.item()
            num_train_steps += 1

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(model, 1e-1)
            optimizer.step()

        avg_training_loss /= num_train_steps

        # Evaluate
        if args.do_eval:
            model.eval()
            y_true = []
            y_pred = []
            paper_ids = []
            avg_val_loss = 0.0
            num_val_steps = 1
            for batch_idx, cur_batch in enumerate(tqdm(val_data_loader, desc="Val Epoch {}".format(epoch))):
                inputs = cur_batch["inputs"].to(device)
                inputs_lengths = cur_batch["inputs_lengths"].to(device)
                targets = cur_batch["targets"].to(device)

                paper_ids += list(cur_batch["paper_ids"])

                with torch.no_grad():
                    predictions, attn_weight = model(inputs, inputs_lengths, batch_size=inputs.shape[0])

                if args.loss_type == "regression":
                    targets = targets.view((-1, 1))
                    loss = loss_criterion(predictions, targets)
                    y_pred += list(predictions.view(targets.size()).data.cpu().tolist())
                    y_true += list(targets.cpu().tolist())
                elif args.loss_type == "classification":
                    targets = targets.type(torch.LongTensor).to(device)
                    loss = loss_criterion(predictions, targets)
                    y_pred += list(torch.max(predictions, 1)[1].view(targets.size()).data.cpu().tolist())
                    y_true += list(targets.cpu().tolist())
                else:
                    raise ValueError("Loss type argument is invalid")

                avg_val_loss += loss.item()
                num_val_steps += 1

            avg_val_loss /= num_val_steps
            metrics = computeMetrics(y_true, y_pred, loss_type=args.loss_type)
            paper_ids_list = paper_ids
            metrics["avg_val_loss"] = avg_val_loss
            print("Metrics :", metrics)

            if args.loss_type == "regression":
                if best_val_loss is None:
                    best_val_loss = metrics["rmse"]
                    best_epoch = epoch
                    best_metrics = metrics
                    best_predictions = y_pred
                elif metrics["rmse"] < best_val_loss:
                    best_val_loss = metrics["rmse"]
                    best_epoch = epoch
                    best_metrics = metrics
                    best_predictions = y_pred
            else:
                if best_val_loss is None:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_metrics = metrics
                    best_predictions = y_pred
                elif avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_metrics = metrics
                    best_predictions = y_pred
        else:
            avg_val_loss = None
            metrics = None

        # Save the model after every epoch
        model_save_path = os.path.join(output_dir, 'model_epoch={}.tar'.format(epoch))
        print("Saving model " + model_save_path)
        torch.save({
            'iteration': epoch,
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
        }, model_save_path)

        # Log the training process
        training_logs.append({
            "epoch": epoch,
            "avg_train_loss": avg_training_loss,
            "avg_val_loss": avg_val_loss,
            "metrics": metrics
        })

    # Write the training logs:
    log_save_path = os.path.join(output_dir, "training_logs.json")
    with open(log_save_path, "w") as f:
        json.dump(training_logs, f)

    # Best results:
    pred_result = []
    if args.do_eval:
        print("Best epoch = ", best_epoch)
        print("Best metrics = ", best_metrics)
        best_metrics["stop_epoch"] = best_epoch + 1

        for cur_paper_id, cur_pred in zip(paper_ids_list, best_predictions):
            if args.loss_type == "regression":
                pred_result.append([cur_paper_id, cur_pred[0]])
            else:
                pred_result.append([cur_paper_id, cur_pred])

    return best_metrics, pred_result

def getTrainTestDatasetSplit(total_dataset, fold_id=-1):
    if fold_id == -1:
        train_dataset = total_dataset
        val_dataset = None
    else:
        train_dataset = []
        val_dataset = []
        for cur_record in total_dataset:
            if int(cur_record["fold_id"]) == fold_id:
                val_dataset.append(cur_record)
            else:
                train_dataset.append(cur_record)

    return train_dataset, val_dataset

def collate_with_padding(batch):
    sorted_batch = sorted(batch, key=lambda x: x["inputs"].shape[0], reverse=True)
    inputs_list = [cur_row["inputs"] for cur_row in sorted_batch]
    inputs_lengths = torch.FloatTensor([len(cur_input) for cur_input in inputs_list])
    targets_list = torch.FloatTensor([cur_row["targets"] for cur_row in sorted_batch])
    inputs_padded_list = nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)
    paper_ids_list = [cur_row["paper_ids"] for cur_row in sorted_batch]

    result_batch = {
        "inputs": inputs_padded_list,
        "inputs_lengths": inputs_lengths,
        "targets": targets_list,
        "paper_ids": paper_ids_list
    }
    return result_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters:
    parser.add_argument("--model_type", type=str, default="M1",
                        help="Model type. M1: Scibert + claim only, M2: Scibert, M3: Longformer")
    parser.add_argument("--loss_type", type=str, default="regression",
                        help="Loss type. Options: 1. regression and 2. classification")
    parser.add_argument("--cv_folds", type=str, default="-1",
                        help="Fold Ids to do CV. Set -1 to use all data for training")

    parser.add_argument("--data_type", type=str, default="TA2",
                        help="Data Type. Options: 1. RPP and 2. TA2")
    parser.add_argument("--data_dir", type=str, default="../data_processed/ta2_folds/fold_1",
                        help="Data directory which contains training data")
    parser.add_argument("--output_dir", type=str, default="scibert_ta2_claim_only",
                        help="Output directory to store the model checkpoints")

    # Optional parameters:
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Train Batch Size")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Eval Batch Size")
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

    total_dataset = REPRDataset(args, model_type=args.model_type, type_path="total")

    # Train on the entire data. No validation is performed
    if args.cv_folds == "-1":
        train_dataset, val_dataset = getTrainTestDatasetSplit(total_dataset, fold_id=-1)

        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_with_padding)
        val_dataloader = None

        if args.loss_type == "regression":
            model = AttentionModel(args.train_batch_size, 1, 100, 768)
        elif args.loss_type == "classification":
            model = AttentionModel(args.train_batch_size, 2, 100, 768)
        else:
            raise ValueError("Loss type argument is invalid")

        # Train the model
        args.do_eval = False
        best_metrics, best_predictions = train(model, train_dataloader, val_dataloader, args, fold_id=-1)

    else: # Do cross validation on the data
        cv_folds = [int(x) for x in args.cv_folds.split(",")]

        # Summary results
        if args.loss_type == "regression":
            metric_vals = ["RMSE"]
        elif args.loss_type == "classification":
            metric_vals = ["Accuracy", "Precision", "Recall", "F1"]
        else:
            raise ValueError("Loss type argument is invalid")

        df_result_list = []
        df_cols_list = ["Fold_Id", "num_train_samples", "num_val_samples", "stop_epoch"] + metric_vals

        # Summary predictions
        df_pred_list = []
        df_pred_cols = ["paper_id", "pred_label"]

        for cur_fold in cv_folds:
            train_dataset, val_dataset = getTrainTestDatasetSplit(total_dataset, fold_id=cur_fold)
            num_train_samples = len(train_dataset)
            num_val_samples = len(val_dataset)
            print("Fold Id = {} ".format(cur_fold), num_train_samples, num_val_samples)

            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_with_padding)
            val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_with_padding)

            if args.loss_type == "regression":
                model = AttentionModel(args.train_batch_size, 1, 100, 768)
            elif args.loss_type == "classification":
                model = AttentionModel(args.train_batch_size, 2, 100, 768)
            else:
                raise ValueError("Loss type argument is invalid")

            # Train the model
            args.do_eval = True
            best_metrics, best_predictions = train(model, train_dataloader, val_dataloader, args, fold_id=cur_fold)

            if args.loss_type == "regression":
                metric_vals_list = [best_metrics["rmse"]]
            elif args.loss_type == "classification":
                metric_vals_list = [best_metrics["accuracy"], best_metrics["precision"], best_metrics["recall"], best_metrics["f1"]]
            else:
                raise ValueError("Loss type argument is invalid")

            df_result_list.append([cur_fold, num_train_samples, num_val_samples, best_metrics["stop_epoch"]] + metric_vals_list)
            df_pred_list += best_predictions


        # Write the summary results
        df_result = pd.DataFrame(data=df_result_list, columns=df_cols_list)
        summary_results_path = os.path.join(args.output_dir, "results_summary.csv")
        df_result.to_csv(summary_results_path, index=False)

        # Write the prediction results
        df_pred = pd.DataFrame(data=df_pred_list, columns=df_pred_cols)
        pred_results_path = os.path.join(args.output_dir, "predictions_summary.csv")
        df_pred.to_csv(pred_results_path, index=False)
