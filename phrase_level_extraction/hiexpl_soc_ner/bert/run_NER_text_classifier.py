import argparse
import os
import pandas as pd
import numpy as np
import re
import torch
import datetime
import time
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import copy
import json

class NERTCDataset(Dataset):
    def __init__(self, args, tokenizer, type_path="test"):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path

        self.input_ids = []
        self.input_mask = []
        self.segment_ids = []
        self.targets = []

        self._build()

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "input_mask":  self.input_mask[index],
            "segment_ids": self.segment_ids[index],
            "targets": self.targets[index],
        }

    def __len__(self):
        return len(self.input_ids)

    def _build(self):
        if self.type_path == "train":
            data_path = os.path.join(self.args.data_dir, "train.txt")
        else:
            data_path = os.path.join(self.args.data_dir, "test.txt")

        with open(data_path, "r") as f:
            for cur_line in f:
                cur_dict = json.loads(cur_line)
                self._create_features(cur_dict["sentence"], cur_dict["entity"], cur_dict["label_idx"])

    def _create_features(self, cur_sentence, cur_entity, cur_label):
        encoded_vals = self.tokenizer.encode_plus(
            text=cur_sentence,
            text_pair=cur_entity,
            add_special_tokens=True,
            max_length=self.args.max_seq_length,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        self.input_ids.append(encoded_vals["input_ids"].squeeze())
        self.segment_ids.append(encoded_vals["token_type_ids"].squeeze())
        self.input_mask.append(encoded_vals["attention_mask"].squeeze())
        self.targets.append(cur_label)


def format_time(elapsed_seconds):
    elapsed_rounded = int(round((elapsed_seconds)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def computeMetrics(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred, average="micro")

    # Convert to labels:
    label_list = ['SP', 'TE', 'SD', 'ES', 'PR', 'PV', 'SS', 'TN']
    idx2label = {}
    for i, label in enumerate(label_list):
        idx2label[i] = label

    y_pred = [idx2label[x] for x in y_pred]
    y_true = [idx2label[x] for x in y_true]

    print(classification_report(y_true, y_pred))
    return accuracy, f1_val

def saveModel(model, model_suffix, args):
    print("Saving model = ", "model" + str(model_suffix))
    # Save the model:
    model.save_pretrained(os.path.join(args.output_dir, "model" + str(model_suffix)))

    # Save the model configuration:
    with open(os.path.join(args.output_dir, "model" + str(model_suffix) + "_params.json"), "w") as f:
        json.dump(vars(args), f)

def train(model, train_data_loader, val_data_loader, args):
    # Get the device info:
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, eps=1e-8)

    total_training_steps = len(train_data_loader) * args.num_epochs
    warmup_steps = int(total_training_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_training_steps)

    training_logs = []
    inital_time = time.time()
    minimum_val_loss = None
    best_model_state_dict = None
    best_model_epoch = None
    for cur_epoch in tqdm(range(args.num_epochs)):
        print("\n\nEpoch number = ", cur_epoch, "\n")

        print("\n\nTraining...")
        start_time = time.time()
        total_train_loss = 0

        model.train()
        for cur_step, cur_batch in enumerate(tqdm(train_data_loader)):
            cur_input_ids = cur_batch["input_ids"].to(device)
            cur_attention_masks = cur_batch["input_mask"].to(device)
            cur_segment_ids = cur_batch["segment_ids"].to(device)
            cur_labels = cur_batch["targets"].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids=cur_input_ids,
                            attention_mask=cur_attention_masks,
                            token_type_ids=cur_segment_ids,
                            labels=cur_labels)
            total_train_loss += loss.item()

            # Backprop the loss
            loss.backward()

            # Perform Gradient Clipping to 1.0:
            clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters:
            optimizer.step()

            # Update learning rate:
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_data_loader)
        total_train_time = format_time(time.time() - start_time)
        print("Training time for Epoch ", cur_epoch, " = ", total_train_time)
        print("Average training loss = ", avg_train_loss)


        print("\n\nValidation...")
        start_time = time.time()
        total_val_loss = 0
        y_preds_total = []
        y_labels_total = []

        # Setting the model in eval mode:
        model.eval()
        for cur_step, cur_batch in enumerate(tqdm(val_data_loader)):
            cur_input_ids = cur_batch["input_ids"].to(device)
            cur_attention_masks = cur_batch["input_mask"].to(device)
            cur_segment_ids = cur_batch["segment_ids"].to(device)
            cur_labels = cur_batch["targets"].to(device)

            with torch.no_grad():
                cur_loss, cur_logits = model(input_ids=cur_input_ids,
                                        attention_mask=cur_attention_masks,
                                        token_type_ids=cur_segment_ids,
                                        labels=cur_labels)

                total_val_loss += cur_loss.item()
                cur_logits_cpu = cur_logits.detach().cpu().numpy()
                cur_labels_cpu = cur_labels.to("cpu").numpy()

                y_preds_flat = list(np.argmax(cur_logits_cpu, axis=1).flatten())
                y_labels_flat = list(cur_labels_cpu.flatten())

                y_preds_total += y_preds_flat
                y_labels_total += y_labels_flat

        avg_val_loss = total_val_loss / len(val_data_loader)
        val_accuracy, val_f1_score = computeMetrics(y_preds_total, y_labels_total)
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
            saveModel(model, "_best", args)
        elif avg_val_loss < minimum_val_loss:
            minimum_val_loss = avg_val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_model_epoch = cur_epoch
            saveModel(model, "_best", args)

        training_logs.append(
        {
                "epoch": cur_epoch,
                "avg_train_loss": avg_train_loss,
                "train_time": total_train_time,
                "avg_val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1_score,
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

    parser.add_argument('--data_dir', type=str, default="../data/ner_dataset_tc",
                        help='Path for Data files')
    parser.add_argument('--output_dir', type=str, default="../models/repr_bert_ner_tc",
                        help='Path to save the checkpoints')
    parser.add_argument('--model_name_or_path', type=str, default="allenai/scibert_scivocab_uncased",
                        help='Model name or Path')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="allenai/scibert_scivocab_uncased",
                        help='Tokenizer name or Path')

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_output_classes', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)


    args = parser.parse_known_args()[0]
    print(args)

    # Get the device info:
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)

    train_dataset = NERTCDataset(args, tokenizer, type_path="train")
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)

    test_dataset = NERTCDataset(args, tokenizer, type_path="test")
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size)

    print(len(train_dataset), len(test_dataset))

    # Model training:
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                num_labels=args.num_output_classes,
                                                                output_attentions=False,
                                                                output_hidden_states=False)
    model.to(device)
    train(model, train_data_loader, test_data_loader, args)