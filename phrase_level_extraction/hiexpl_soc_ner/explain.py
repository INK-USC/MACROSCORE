# from algo.soc_lstm import SOCForLSTM
from algo.soc_transformer import SOCForTransformer
import torch
import argparse
from utils.args import get_args
from utils.reader import load_vocab
from utils.reader import get_data_iterators_repr
import random, os
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertForTokenClassification
import json
from tqdm import tqdm

def get_args_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    args = parser.parse_args()
    return args

args = get_args()

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

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    if args.task == 'repr':
        args.vocab_path = "vocab/vocab_repr_claims_bert.pkl"
        tree_path = ""
    elif args.task == 'repr_ner_tc':
        args.vocab_path = "vocab/vocab_repr_claims_bert.pkl"
        tree_path = "data/ner_dataset_tc/trees/{}.jsonl".format(args.dataset)
    elif args.task == 'repr_ner_seq':
        args.vocab_path = "vocab/vocab_repr_claims_bert.pkl"
        tree_path = "data/ner_dataset_tc/trees/{}.jsonl".format(args.dataset)
    else:
        raise ValueError('unknown task')

    vocab = load_vocab(args.vocab_path)

    args.n_embed = len(vocab)
    args.d_out = 8  # Number of classes
    args.n_cells = args.n_layers
    args.use_gpu = args.gpu >= 0

    if args.explain_model == 'bert':
        if args.task == "repr_ner_tc":
            WEIGHTS_NAME = 'pytorch_model.bin'
            output_model_file = os.path.join('%s' % args.resume_snapshot, WEIGHTS_NAME)

            # Load a trained model and config that you have fine-tuned
            model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=args.d_out)
            model.load_state_dict(torch.load(output_model_file))
            model.eval()
        elif args.task == "repr_ner_seq":
            WEIGHTS_NAME = 'pytorch_model.bin'
            CONFIG_FILE = "config.json"
            output_model_file = os.path.join('%s' % args.resume_snapshot, WEIGHTS_NAME)
            output_config_file = os.path.join('%s' % args.resume_snapshot, CONFIG_FILE)

            # Load a trained model and config that you have fine-tuned
            config = BertConfig.from_pretrained(output_config_file)
            model = BertForNER.from_pretrained(args.bert_model, config=config)
            model.load_state_dict(torch.load(output_model_file))
            model.eval()

        if args.gpu >= 0:
            model = model.to(args.gpu)
        if args.method == 'soc':
            lm_model = torch.load(args.lm_path, map_location=lambda storage, location: storage.cuda(args.gpu))
            lm_model.gpu = args.gpu
            lm_model.encoder.gpu = args.gpu
            algo = SOCForTransformer(model, lm_model,
                                     tree_path=None,
                                     output_path='outputs/' + args.task,
                                     config=args, vocab=vocab)
        else:
            raise ValueError('unknown method')
    else:
        raise ValueError('unknown model')

    tree_data = []
    with open(tree_path, "r") as f:
        for cur_line in f:
            tree_data.append(json.loads(cur_line.strip()))

    target_entity_class = args.target_entity_class
    filtered_tree_data = []
    for cur_record in tree_data:
        if cur_record["label"] == target_entity_class:
            filtered_tree_data.append(cur_record)

    print("Number of records with label {} = {}".format(target_entity_class, len(filtered_tree_data)))
    results_all = []
    for cur_idx in tqdm(range(len(filtered_tree_data))):
        cur_record = filtered_tree_data[cur_idx]
        if args.task == "repr_ner_tc":
            cur_result = algo.explain_repr_ner_tc_single(cur_record, topk=3)
        elif args.task == "repr_ner_seq":
            cur_result = algo.explain_repr_ner_seq_single(cur_record, topk=3)
        else:
            raise Exception("Invalid NER task type")

        if cur_result is None:
            print("Error for ", cur_idx)
        else:
            cur_record["predicted_label"] = cur_result["predicted_label"]
            cur_record["important_phrases"] = cur_result["important_phrases"]
            results_all.append(cur_record)
        break

    out_path = 'outputs/' + args.task + "_" + args.target_entity_class + ".json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)

