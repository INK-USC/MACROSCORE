from algo.soc_lstm import *
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from transformers import BertForPreTraining, BertTokenizer, BertForTokenClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.parser import get_span_to_node_mapping, read_trees_from_corpus_repr, parse_tree
from utils.reader import get_examples_repr_from_trees, get_parsed_example_repr_ner_from_tree_string, get_parsed_example_repr_ner_from_text
import numpy as np
import copy
import pickle
import json
from utils.args import get_args

DotDict = Batch
args = get_args()


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = example.text
        mapping = example.mapping
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        mapping += [-1] * (max_seq_length - len(mapping))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

        features.append(
            DotDict(input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    offset=example.offset,
                    mapping=mapping))
    return features


def convert_examples_to_features_with_textpair(example, input_record, max_seq_length, tokenizer):
    tokens_a = example.text
    tokens_b = tokenizer.tokenize(input_record["entity"])
    mapping = example.mapping
    if len(tokens_a) > max_seq_length - 3 - len(tokens_b):
        tokens_a = tokens_a[:(max_seq_length - 3 - len(tokens_b))]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    mapping += [-1] * (max_seq_length - len(mapping))

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = example.label

    features = DotDict(input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                offset=example.offset,
                mapping=mapping)
    return features


def bert_id_to_lm_id(arr, bert_tokenizer, lm_vocab):
    tmp_tokens = bert_tokenizer.convert_ids_to_tokens(arr)
    tokens = []
    conv_dict = {'[UNK]': '<unk>', '[PAD]': '<pad>'}
    for w in tmp_tokens:
        tokens.append(conv_dict.get(w, w))
    lm_ids = [lm_vocab.stoi.get(token, 0) for token in tokens]
    return np.array(lm_ids, dtype=np.int32)


def lm_id_to_bert_id(arr, bert_tokenizer, lm_vocab):
    tmp_tokens = [lm_vocab.itos[x] for x in arr.tolist()]
    tokens = []
    conv_dict = {'<unk>': '[UNK]', '<pad>': '[PAD]'}
    for w in tmp_tokens:
        tokens.append(conv_dict.get(w, w))
    bert_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    return np.array(bert_ids, dtype=np.int32)


def get_data_iterator_bert(tree_path, tokenizer, max_seq_length, batch_size, label_vocab=None):
    if args.task == 'repr':
        eval_examples = get_examples_repr_from_trees(tree_path, train_lm=False, bert_tokenizer=tokenizer)
    else:
        raise ValueError
    eval_features = convert_examples_to_features(eval_examples, max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_offsets = torch.tensor([f.offset for f in eval_features], dtype=torch.long)
    all_mappings = torch.tensor([f.mapping for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_offsets, all_mappings)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    return eval_dataloader


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


class NERInferenceModule:
    def __init__(self, tokenizer, model, max_seq_length):
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_length = max_seq_length

        self.label2idx = {}
        self.idx2label = {}
        self._create_label_map()

    def pre_process_input(self, input_examples_list):

        features_list = []
        for cur_input_example in input_examples_list:
            tokens = cur_input_example.text
            mapping = cur_input_example.mapping
            valid_positions = [0] * len(tokens)

            # Make start as valid and use the mapping to construct the valid_positions
            valid_positions[0] = 1
            for lm_idx in mapping:
                if lm_idx < len(valid_positions):
                    valid_positions[lm_idx] = 1

            if len(tokens) >= self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
                valid_positions = valid_positions[0:(self.max_seq_length - 2)]

            # Insert "[CLS]"
            tokens.insert(0, "[CLS]")
            valid_positions.insert(0, 1)

            # Insert "[SEP]"
            tokens.append("[SEP]")
            valid_positions.append(1)

            # Get ids
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Pad till the max_seq_length
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid_positions.append(1)

            # Fill mapping
            mapping += [-1] * (self.max_seq_length - len(mapping))

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(valid_positions) == self.max_seq_length
            assert len(mapping) == self.max_seq_length

            features_list.append({
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "valid_positions": valid_positions,
                "mapping": mapping
            })

        return features_list

    def predict(self, input_ids, input_mask, segment_ids, valid_positions, lm_entity_span):
        with torch.no_grad():
            logits = self.model(input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                valid_ids=valid_positions)
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()  # Batch size X Max Seq length

        # Move one index right to account for CLS token at the beginning
        lm_entity_span = [x+1 for x in lm_entity_span]

        predictions = []
        for i in range(len(logits)):
            cur_pred = []
            for j in lm_entity_span:
                cur_pred.append(self.idx2label[logits[i][j]])

            tag_name = None
            flag = 1
            for j in cur_pred:
                try:
                    cur_tag_name = j.split("-")[-1]
                    if tag_name is None:
                        tag_name = cur_tag_name
                    elif tag_name != cur_tag_name:
                        flag = 0
                except:
                    flag = 0

            if flag == 0:
                tag_name = "INVALID:" + ",".join(cur_pred)

            predictions.append(tag_name)

        return predictions

    def getScore(self, input_ids, input_mask, segment_ids, valid_positions, lm_entity_span, gt_label):
        with torch.no_grad():
            logits = self.model(input_ids=input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                valid_ids=valid_positions)
        logits = logits.detach().cpu().numpy()  # Batch size X Max Seq length X Num classes

        # Move one index right to account for CLS token at the beginning
        lm_entity_span = [x + 1 for x in lm_entity_span]

        # Get the index of the gt_label
        b_idx = self.label2idx["B-" + gt_label]
        i_idx = self.label2idx["I-" + gt_label]

        logits_reduced = []
        for i in range(len(logits)):
            start = True
            cur_logit_reduced = []
            for j in lm_entity_span:
                if start:
                    cur_logit_reduced.append(logits[i][j][b_idx])
                    start = False
                else:
                    cur_logit_reduced.append(logits[i][j][i_idx])
            logits_reduced.append(cur_logit_reduced)

        logits_reduced = np.array(logits_reduced)  # Batch size X Entity span length

        return logits_reduced

    def _create_label_map(self):
        # Create the label map
        label_set = ['[PAD]', 'O', 'B-ES', 'I-ES', 'B-PR', 'I-PR', 'B-PV', 'I-PV',
                     'B-SD', 'I-SD', 'B-SP', 'I-SP', 'B-SS', 'I-SS', 'B-TE',
                     'I-TE', 'B-TN', 'I-TN', '[CLS]', '[SEP]']
        # label_set = ["[PAD]", "O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        for idx, val in enumerate(label_set):
            self.label2idx[val] = idx
            self.idx2label[idx] = val

    def get_label_map(self):
        return self.label2idx, self.idx2label


class ExplanationBaseForTransformer(ExplanationBase):
    def __init__(self, model, tree_path, output_path, config):
        super().__init__(model, None, None, tree_path, output_path, config)
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='bert/cache')
        self.tree_path = tree_path
        self.max_seq_length = 128
        self.batch_size = config.batch_size
        self.iterator = self.get_data_iterator()

    def get_data_iterator(self):
        return get_data_iterator_bert(self.tree_path, self.tokenizer, self.max_seq_length, self.batch_size)

    def occlude_input_with_masks(self, inp, inp_mask, x_regions, nb_regions):
        region_indicator = np.zeros(len(inp), dtype=np.int32)
        for region in nb_regions:
            region_indicator[region[0]:region[1] + 1] = 2
        for region in x_regions:
            region_indicator[region[0]:region[1] + 1] = 1
        # input expectation over neighbourhood
        inp_enb = copy.copy(inp)
        inp_mask_enb = copy.copy(inp_mask)
        # input expectation over neighbourhood and selected span
        inp_ex = copy.copy(inp)
        inp_mask_ex = copy.copy(inp_mask)

        inp_enb, inp_mask_enb = self.mask_region_masked(inp_enb, inp_mask_enb, region_indicator, [2])
        inp_ex, inp_mask_ex = self.mask_region_masked(inp_ex, inp_mask_ex, region_indicator, [1, 2])

        return inp_enb, inp_mask_enb, inp_ex, inp_mask_ex

    def mask_region_masked(self, inp, inp_mask, region_indicator, mask_value):
        new_seq = []
        new_mask_seq = []
        for i in range(len(region_indicator)):
            if region_indicator[i] not in mask_value:
                new_seq.append(inp[i])
            else:
                new_seq.append(self.tokenizer.vocab['[PAD]'])
            new_mask_seq.append(inp_mask[i])
        if not new_seq:
            new_seq.append(self.tokenizer.vocab['[PAD]'])
        new_seq = np.array(new_seq)
        new_mask_seq = np.array(new_mask_seq)
        return new_seq, new_mask_seq

    def get_ngram_mask_region(self, region, inp):
        # find the [PAD] token
        idx = 0
        while idx < len(inp) and inp[idx] != 0:
            idx += 1
        if not self.nb_unidirectional:
            return [(max(region[0] - self.nb_range, 1), min(region[1] + self.nb_range, idx - 2))]
        else:
            return [(max(1, region[0]), min(region[1] + self.nb_range, idx - 2))] if not args.task == 'tacred' else \
                [(max(0, region[0]), min(region[1] + self.nb_range, idx - 1))]

    def explain_single_transformer(self, input_ids, input_mask, segment_ids, region, label=None):

        inp_flatten = input_ids.view(-1).cpu().numpy()
        inp_mask_flatten = input_mask.view(-1).cpu().numpy()
        mask_regions = self.get_ngram_mask_region(region, inp_flatten)

        inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = self.occlude_input_with_masks(inp_flatten, inp_mask_flatten,
                                                                                   [region], mask_regions)

        inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = torch.from_numpy(inp_enb).long().view(1, -1), torch.from_numpy(
            inp_mask_enb).long().view(1, -1), torch.from_numpy(inp_ex).long().view(1, -1), torch.from_numpy(
            inp_mask_ex).long().view(1, -1)

        if self.gpu >= 0:
            inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = inp_enb.to(self.gpu), inp_mask_enb.to(self.gpu), \
                                                         inp_ex.to(self.gpu), inp_mask_ex.to(self.gpu)
            segment_ids = segment_ids.to(self.gpu)
        logits_enb = self.model(inp_enb, segment_ids[:, :inp_enb.size(1)], inp_mask_enb)
        logits_ex = self.model(inp_ex, segment_ids[:, :inp_ex.size(1)], inp_mask_ex)
        contrib_logits = logits_enb - logits_ex  # [1 * C]
        if contrib_logits.size(1) == 2:
            contrib_score = contrib_logits[0, 1] - contrib_logits[0, 0]
        else:
            contrib_score = normalize_logit(contrib_logits, label)
        return contrib_score.item()

    def agglomerate(self, inputs, percentile_include, method, sweep_dim, dataset,
                    num_iters=5, subtract=True, absolute=True, label=None):
        text_orig = inputs[0].cpu().clone().numpy().transpose((1, 0))
        for t in range(text_orig.shape[0]):
            if text_orig[t, 0] == 0:
                text_orig = text_orig[:t]
                break
        text_len = text_orig.shape[0]
        score_orig = self.explain_single_transformer(*inputs, region=[1, text_len - 2], label=label)
        # get scores
        texts = gen_tiles(text_orig, method=method, sweep_dim=sweep_dim)
        texts = texts.transpose()

        starts, stops = tiles_to_cd(texts)

        scores = []
        for (start, stop) in zip(starts, stops):
            # filter the case when start covers CLS
            if start == 0:
                start = 1
            # filter the case when stop covers SEP
            if stop == text_len - 1:
                stop = text_len - 2
            # if start > stop, then return 0 (e.g. score for word CLS, SEP)
            if start > stop:
                score = 0
            else:
                score = self.explain_single_transformer(*inputs, region=[start, stop], label=label)
            scores.append(score)

        # threshold scores
        mask = threshold_scores(scores, percentile_include, absolute=absolute)

        # initialize lists
        scores_list = [np.copy(scores)]
        mask_list = [mask]
        comps_list = []
        comp_scores_list = [{0: score_orig}]

        # iterate
        for step in range(num_iters):
            # find connected components for regions
            comps = np.copy(measure.label(mask_list[-1], background=0, connectivity=1))

            # loop over components
            comp_scores_dict = {}
            for comp_num in range(1, np.max(comps) + 1):

                # make component tile
                comp_tile_bool = (comps == comp_num)
                comp_tile = gen_tile_from_comp(text_orig, comp_tile_bool, method)

                # make tiles around component
                border_tiles = gen_tiles_around_baseline(text_orig, comp_tile_bool,
                                                         method=method,
                                                         sweep_dim=sweep_dim)

                # predict for all tiles
                # format tiles into batch
                tiles_concat = np.hstack((comp_tile, np.squeeze(border_tiles[0]).transpose()))

                starts, stops = tiles_to_cd(tiles_concat)
                scores_all = []
                for (start, stop) in zip(starts, stops):
                    # filter the case when start covers CLS
                    if start == 0:
                        start = 1
                    # filter the case when stop covers SEP
                    if stop == text_len - 1:
                        stop = text_len - 2
                    # if start > stop, then return 0 (e.g. score for word CLS, SEP)
                    if start > stop:
                        score = 0
                    else:
                        score = self.explain_single_transformer(*inputs, region=[start, stop], label=label)
                    scores_all.append(score)

                score_comp = np.copy(scores_all[0])
                scores_border_tiles = np.copy(scores_all[1:])

                # store the predicted class scores
                comp_scores_dict[comp_num] = np.copy(score_comp)

                # update pixel scores
                tiles_idxs = border_tiles[1]
                for i, idx in enumerate(tiles_idxs):
                    scores[idx] = scores_border_tiles[i] - score_comp

            # get class preds and thresholded image
            scores = np.array(scores)
            scores[mask_list[-1]] = np.nan
            mask = threshold_scores(scores, percentile_include, absolute=absolute)

            # add to lists
            scores_list.append(np.copy(scores))
            mask_list.append(mask_list[-1] + mask)
            comps_list.append(comps)
            comp_scores_list.append(comp_scores_dict)

            if np.sum(mask) == 0:
                break

        # pad first image
        comps_list = [np.zeros(text_orig.size, dtype=np.int)] + comps_list

        return {'scores_list': scores_list,  # arrs of scores (nan for selected)
                'mask_list': mask_list,  # boolean arrs of selected
                'comps_list': comps_list,  # arrs of comps with diff number for each comp
                'comp_scores_list': comp_scores_list,  # dicts with score for each comp
                'score_orig': score_orig}  # original score

    def map_lm_to_bert_token(self, lm_idx, mapping):
        left = 0 if lm_idx == 0 else mapping[lm_idx - 1]
        right = mapping[lm_idx] - 1
        return left, right

    def formatPhrases(self, inp_string, do_reverse=False):
        inp_string_split = inp_string.split("\t")
        out_list = []

        for cur_phrase in inp_string_split:
            cur_phrase_split = cur_phrase.split()
            if len(cur_phrase_split) < 2:
                continue
            phrase_tokens = cur_phrase_split[:-1]
            score = float(cur_phrase_split[-1])

            phrase_string = ""
            for cur_token in phrase_tokens:
                if cur_token.startswith("##"):
                    phrase_string += cur_token[2:]
                else:
                    phrase_string += " " + cur_token
            phrase_string = phrase_string.strip()

            phrase_string = phrase_string.replace("- lrb -", "(")
            phrase_string = phrase_string.replace("- rrb -", ")")
            out_list.append([score, phrase_string])

        out_list = list(sorted(out_list, key=lambda x: x[0], reverse=do_reverse))
        return out_list

    def formatResultDict(self, result_dict_list, do_reverse=False):
        out_list = []

        for cur_dict in result_dict_list:
            phrase_tokens = cur_dict["phrase"].split()
            score = float(cur_dict["score"])
            lm_span = cur_dict["lm_span"]

            phrase_string = ""
            for cur_token in phrase_tokens:
                if cur_token.startswith("##"):
                    phrase_string += cur_token[2:]
                else:
                    phrase_string += " " + cur_token
            phrase_string = phrase_string.strip()

            phrase_string = phrase_string.replace("- lrb -", "(")
            phrase_string = phrase_string.replace("- rrb -", ")")
            out_list.append({
                "score": score,
                "phrase": phrase_string,
                "lm_span": lm_span
            })

        out_list = list(sorted(out_list, key=lambda x: x["score"], reverse=do_reverse))
        return out_list

    def explain_repr_ner_tc_single_custom(self, input_record, topk=3):
        label_idx = input_record["label_idx"]
        input_sentence = input_record["sentence"]
        input_sentence_tokens = input_sentence.split()

        # Convert inputs to features:
        input_example = get_parsed_example_repr_ner_from_text(input_sentence, self.tokenizer)
        input_features = convert_examples_to_features_with_textpair(input_example, input_record, self.max_seq_length, self.tokenizer)
        input_ids = torch.tensor(input_features.input_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(input_features.input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(input_features.segment_ids, dtype=torch.long).unsqueeze(0)
        input_mappings = torch.tensor(input_features.mapping, dtype=torch.long).unsqueeze(0)

        try:
            logits = self.model(input_ids=input_ids.cuda(), attention_mask=input_mask.cuda())[0]
            _, pred = logits.max(-1)
            pred_label = pred.item()

            inp = input_ids.view(-1).cpu().numpy()
            mappings = input_mappings.view(-1).cpu().numpy()

            spans = []
            for cur_cand in input_record["candidates"]:
                spans.append((cur_cand["candidate_span"][0], cur_cand["candidate_span"][1]))

            repr_spans = []
            contribs = []
            result_dict_list = []
            for span in spans:
                if type(span) is int:
                    span = (span, span)
                bert_span = self.map_lm_to_bert_token(span[0], mappings)[0], \
                            self.map_lm_to_bert_token(span[1], mappings)[1]
                # add 1 to spans since transformer inputs has [CLS]
                repr_spans.append(bert_span)
                bert_span = (bert_span[0] + 1, bert_span[1] + 1)
                contrib = self.explain_single_transformer(input_ids, input_mask, segment_ids, bert_span, label=label_idx)
                contribs.append(contrib)
                result_dict_list.append({
                    "phrase": " ".join(input_sentence_tokens[span[0]: span[-1] + 1]),
                    "score": contrib,
                    "lm_span": span
                })

            # # Post process the results
            # result_dict_list = self.repr_result_post_process(inp, repr_spans, spans, contribs)

            cur_json_dict = {}
            cur_json_dict["predicted_label"] = int(pred_label)
            cur_json_dict["ground_truth_label"] = label_idx
            if cur_json_dict["predicted_label"] == label_idx:
                cur_json_dict["important_phrases"] = self.formatResultDict(result_dict_list, do_reverse=True)
            else:
                cur_json_dict["important_phrases"] = self.formatResultDict(result_dict_list, do_reverse=False)
        except Exception as e:
            print(e)
            cur_json_dict = None

        return cur_json_dict

    def explain_repr_ner_seq_single_custom(self, input_record, topk=3):
        gt_label = input_record["label"]
        input_sentence = input_record["sentence"]
        input_sentence_tokens = input_sentence.split()

        if torch.cuda.device_count() > 0:
            device = "cuda"
        else:
            device = "cpu"

        # Convert inputs to features:
        input_examples = [get_parsed_example_repr_ner_from_text(input_sentence, self.tokenizer)]
        features = self.NER_infer_module.pre_process_input(input_examples)[0]

        input_ids = torch.tensor(features["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        input_mask = torch.tensor(features["input_mask"], dtype=torch.long, device=device).unsqueeze(0)
        segment_ids = torch.tensor(features["segment_ids"], dtype=torch.long, device=device).unsqueeze(0)
        valid_positions = torch.tensor(features["valid_positions"], dtype=torch.long, device=device).unsqueeze(0)
        mapping = torch.tensor(features["mapping"], dtype=torch.long, device=device).unsqueeze(0)

        try:
            pred_label = self.NER_infer_module.predict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                                      valid_positions=valid_positions,
                                                      lm_entity_span=input_record["entity_span"])[0]

            inp = input_ids.view(-1).cpu().numpy()
            mappings = mapping.view(-1).cpu().numpy()

            spans = []
            for cur_cand in input_record["candidates"]:
                spans.append((cur_cand["candidate_span"][0], cur_cand["candidate_span"][1]))

            repr_spans = []
            contribs = []
            result_dict_list = []
            for span in spans:
                if type(span) is int:
                    span = (span, span)
                bert_span = self.map_lm_to_bert_token(span[0], mappings)[0], \
                            self.map_lm_to_bert_token(span[1], mappings)[1]
                # add 1 to spans since transformer inputs has [CLS]
                repr_spans.append(bert_span)
                bert_span = (bert_span[0] + 1, bert_span[1] + 1)
                contrib = self.explain_single_transformer_seq(input_ids, input_mask, segment_ids, valid_positions, bert_span,
                                                              lm_entity_span=input_record["entity_span"], label=gt_label)
                contribs.append(contrib)
                result_dict_list.append({
                    "phrase": " ".join(input_sentence_tokens[span[0]: span[-1]+1]),
                    "score": contrib,
                    "lm_span": span
                })

            # # Post process the results
            # result_dict_list = self.repr_result_post_process(inp, repr_spans, spans, contribs)

            cur_json_dict = {}
            cur_json_dict["predicted_label"] = pred_label
            cur_json_dict["ground_truth_label"] = gt_label
            if cur_json_dict["predicted_label"] == gt_label:
                cur_json_dict["important_phrases"] = self.formatResultDict(result_dict_list, do_reverse=True)
            else:
                cur_json_dict["important_phrases"] = self.formatResultDict(result_dict_list, do_reverse=False)

        except Exception as e:
            print(e)
            cur_json_dict = None

        return cur_json_dict

    def explain_repr(self):
        output_json_path = self.output_path.split(".")[0] + ".json"
        f = open(self.output_path, 'w')

        json_output = []
        all_contribs = []
        cnt = 0
        for batch_idx, (input_ids, input_mask, segment_ids, label_ids, offsets, mappings) in enumerate(self.iterator):
            if batch_idx < self.batch_start:
                continue

            try:
                logits = self.model(input_ids=input_ids.cuda(), attention_mask=input_mask.cuda())[0]
                _, pred = logits.max(-1)
                pred_label = pred.item()

                inp = input_ids.view(-1).cpu().numpy()
                inp_id = offsets.item()
                mappings = mappings.view(-1).cpu().numpy()
                span2node, node2span = get_span_to_node_mapping(self.trees[inp_id])
                spans = list(span2node.keys())
                repr_spans = []
                contribs = []
                for span in spans:
                    if type(span) is int:
                        span = (span, span)
                    bert_span = self.map_lm_to_bert_token(span[0], mappings)[0], \
                                self.map_lm_to_bert_token(span[1], mappings)[1]
                    # add 1 to spans since transformer inputs has [CLS]
                    repr_spans.append(bert_span)
                    bert_span = (bert_span[0] + 1, bert_span[1] + 1)
                    contrib = self.explain_single_transformer(input_ids, input_mask, segment_ids, bert_span)
                    contribs.append(contrib)
                all_contribs.append(contribs)
                s = self.repr_result_region(inp, repr_spans, contribs)
            except:
                print('Error for  %d' % batch_idx)
                continue

            # Write to text file
            f.write(s + '\n')

            # Write to json
            cur_json_dict = {}
            cur_json_dict["paper_id"] = self.paper_info[inp_id]["paper_id"]
            cur_json_dict["label"] = int(self.paper_info[inp_id]["label"])
            cur_json_dict["predicted_label"] = int(pred_label)
            cur_json_dict["important_segment"] = self.paper_info[inp_id]["important_segment"]

            if cur_json_dict["label"] == 1:
                cur_json_dict["important_phrases"] = self.formatPhrases(s, do_reverse=True)
            else:
                cur_json_dict["important_phrases"] = self.formatPhrases(s, do_reverse=False)

            json_output.append(cur_json_dict)
            print('finished %d' % batch_idx)
            cnt += 1
            if batch_idx == self.batch_stop - 1:
                break
        f.close()

        # Write the json output
        with open(output_json_path, "w") as f:
            json.dump(json_output, f, indent=2)

        return all_contribs

    def explain_sst(self):
        f = open(self.output_path, 'w')
        all_contribs = []
        cnt = 0
        for batch_idx, (input_ids, input_mask, segment_ids, label_ids, offsets, mappings) in enumerate(self.iterator):
            if batch_idx < self.batch_start:
                continue

            inp = input_ids.view(-1).cpu().numpy()
            inp_id = offsets.item()
            mappings = mappings.view(-1).cpu().numpy()
            span2node, node2span = get_span_to_node_mapping(self.trees[inp_id])
            spans = list(span2node.keys())
            repr_spans = []
            contribs = []
            for span in spans:
                if type(span) is int:
                    span = (span, span)
                bert_span = self.map_lm_to_bert_token(span[0], mappings)[0], \
                            self.map_lm_to_bert_token(span[1], mappings)[1]
                # add 1 to spans since transformer inputs has [CLS]
                repr_spans.append(bert_span)
                bert_span = (bert_span[0] + 1, bert_span[1] + 1)
                contrib = self.explain_single_transformer(input_ids, input_mask, segment_ids, bert_span)
                contribs.append(contrib)
            all_contribs.append(contribs)

            s = self.repr_result_region(inp, repr_spans, contribs)
            f.write(s + '\n')

            print('finished %d' % batch_idx)
            cnt += 1
            if batch_idx == self.batch_stop - 1:
                break
        f.close()
        return all_contribs

    def explain_agg(self, dataset):
        f = open(self.output_path.replace('.txt','.pkl'), 'wb')
        all_tabs = []

        for batch_idx, (input_ids, input_mask, segment_ids, label_ids, offsets, mappings) in enumerate(self.iterator):
            if batch_idx < self.batch_start:
                continue
            # get prediction
            if label_ids.item() == 0:
                continue
            # segment_ids.cuda()
            logits = self.model(input_ids=input_ids.cuda(), attention_mask=input_mask.cuda())[0]
            _, pred = logits.max(-1)

            if pred.item() != label_ids.item():
                continue

            inp = input_ids.view(-1).cpu().numpy()

            lists = self.agglomerate((input_ids, input_mask, segment_ids), percentile_include=90, method='cd',
                                     sweep_dim=1, dataset=dataset, num_iters=10, label=label_ids.item())
            lists = collapse_tree(lists)
            seq_len = lists['scores_list'][0].shape[0]
            data = lists_to_tabs(lists, seq_len)
            text = ' '.join(self.tokenizer.convert_ids_to_tokens(inp)[:seq_len])
            label_name = self.label_vocab.itos[label_ids.item()] if self.label_vocab is not None else label_ids.item()
            all_tabs.append({
                'tab': data,
                'text': text,
                'label': label_name,
                'pred': normalize_logit(logits, label_ids).item() if logits.size(1) != 2 else
                        logits[:, 1] - logits[:, 0] # [B]
                 })
            print(all_tabs)
            print('finished %d' % batch_idx)

            if batch_idx >= self.batch_stop - 1:
                break
        pickle.dump(all_tabs, f)
        f.close()
        return all_tabs

    def explain_token(self, dataset):
        f = open(self.output_path, 'w')
        all_contribs = []
        cnt = 0
        for batch_idx, (input_ids, input_mask, segment_ids, label_ids, offsets, mappings) in enumerate(self.iterator):
            if batch_idx < self.batch_start:
                continue
            if args.task == 'tacred':
                if label_ids.item() == 0:
                    continue
                logits = self.model(input_ids.cuda(), input_mask.cuda(), segment_ids.cuda())
                _, pred = logits.max(-1)
                if pred.item() != label_ids.item():
                    continue

            inp = input_ids.view(-1).cpu().numpy()
            inp_id = offsets.item()
            mappings = mappings.view(-1).cpu().numpy()

            if 0 in inp.tolist():
                length = inp.tolist().index(0)  # [PAD]
            else:
                length = len(inp)
            repr_spans = []
            contribs = []
            for span in range(length - 2):
                if type(span) is int:
                    span = (span, span)
                bert_span = span
                # add 1 to spans since transformer inputs has [CLS]
                repr_spans.append(bert_span)
                # if not args.task == 'tacred':
                bert_span = (bert_span[0] + 1, bert_span[1] + 1)
                contrib = self.explain_single_transformer(input_ids, input_mask, segment_ids, bert_span,
                                                          label_ids.item())
                contribs.append(contrib)
            all_contribs.append(contribs)

            s = self.repr_result_region(inp, repr_spans, contribs, label=label_ids.item())
            f.write(s + '\n')

            print('finished %d' % batch_idx)
            cnt += 1
            if batch_idx == self.batch_stop - 1:
                break
        f.close()
        return all_contribs

    def map_lm_to_bert_span(self, span, mappings):
        span = self.map_lm_to_bert_token(span[0], mappings)[0], self.map_lm_to_bert_token(span[1], mappings)[1]
        # add 1 to spans since transformer inputs has [CLS]
        span = (span[0] + 1, span[1] + 1)
        return span

    def repr_result_region(self, inp, spans, contribs, label=None):
        tokens = self.tokenizer.convert_ids_to_tokens(inp)
        outputs = []
        assert (len(spans) == len(contribs))
        for span, contrib in zip(spans, contribs):
            outputs.append((' '.join(tokens[span[0] + 1:span[1] + 2]), contrib))
        output_str = ' '.join(['%s %.6f\t' % (x, y) for x, y in outputs])
        if label is not None and hasattr(self, 'label_vocab') and self.label_vocab is not None:
            output_str = self.label_vocab.itos[label] + '\t' + output_str
        return output_str

    def repr_result_post_process(self, inp, spans, lm_spans, contribs, label=None):
        tokens = self.tokenizer.convert_ids_to_tokens(inp)
        outputs = []
        assert (len(spans) == len(contribs)) and (len(spans) == len(lm_spans))
        for lm_span, span, contrib in zip(lm_spans, spans, contribs):
            outputs.append({
                "phrase": ' '.join(tokens[span[0] + 1:span[1] + 2]),
                "score": contrib,
                "lm_span": lm_span
            })

        return outputs

    def demo(self):
        f = open(self.output_path, 'w')
        while True:
            l = input('sentence?')
            inp_word = ['[CLS]'] + l.strip().split() + ['[SEP]']
            inp_word_id = self.tokenizer.convert_tokens_to_ids(inp_word)
            inp = torch.from_numpy(np.array(inp_word_id, dtype=np.int32)).long().view(1, -1)
            input_mask = torch.ones_like(inp).long().view(1, -1)
            segment_ids = torch.zeros_like(inp).view(1, -1).to(self.gpu).long()
            spans = [(x, x) for x in range(0, len(inp_word_id) - 2)]

            contribs = []
            for span in spans:
                span = (span[0] + 1, span[1] + 1)
                contrib = self.explain_single_transformer(inp, input_mask, segment_ids, span)
                contribs.append(contrib)

            s = self.repr_result_region(inp.view(-1).cpu().numpy(), spans, contribs)
            f.write(s + '\n')
            print(s)


class SOCForTransformer(ExplanationBaseForTransformer):
    def __init__(self, target_model, lm_model, vocab, tree_path, output_path, config):
        super().__init__(target_model, tree_path, output_path, config)
        self.model = target_model
        self.lm_model = lm_model
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, cache_dir='bert/cache')
        self.tree_path = tree_path
        self.max_seq_length = 128
        self.batch_size = config.batch_size
        self.sample_num = config.sample_n
        self.vocab = vocab
        self.task = config.task
        self.iterator = self.get_data_iterator()

        if self.task == 'repr':
            self.trees, self.paper_info = read_trees_from_corpus_repr(tree_path)
        elif self.task == "repr_ner_seq":
            self.NER_infer_module = NERInferenceModule(tokenizer=self.tokenizer, model=self.model, max_seq_length=self.max_seq_length)

        self.use_bert_lm = config.use_bert_lm
        self.bert_lm = None
        if self.use_bert_lm:
            self.bert_lm = BertForPreTraining.from_pretrained(config.bert_model, cache_dir='./bert/cache').to(
                self.gpu)

        self.feasible_bert_ids = self.get_feasible_bert_ids()

    def get_feasible_bert_ids(self):
        s = set()
        for w in self.vocab.stoi:
            s.add(self.tokenizer.vocab.get(w, -1))
        return s

    def get_data_iterator(self):
        # return get_data_iterator_bert(self.tree_path, self.tokenizer, self.max_seq_length, self.batch_size)
        return None

    def score(self, inp_bak, inp_mask, segment_ids, x_regions, nb_regions, label):
        x_region = x_regions[0]
        nb_region = nb_regions[0]
        inp = copy.copy(inp_bak)

        inp_length = 0
        for i in range(len(inp_mask)):
            if inp_mask[i] == 1:
                inp_length += 1
            else:
                break
        # mask everything outside the x_region and inside nb region
        inp_lm = copy.copy(inp)
        for i in range(len(inp_lm)):
            if nb_region[0] <= i <= nb_region[1] and not x_region[0] <= i <= x_region[1]:
                inp_lm[i] = self.tokenizer.vocab['[PAD]']

        inp_th = torch.from_numpy(bert_id_to_lm_id(inp_lm[1:inp_length - 1], self.tokenizer, self.vocab)).long().view(
            -1, 1)
        inp_length = torch.LongTensor([inp_length - 2])
        fw_pos = torch.LongTensor([min(x_region[1] + 1 - 1, len(inp) - 2)])
        bw_pos = torch.LongTensor([max(x_region[0] - 1 - 1, -1)])

        if self.gpu >= 0:
            inp_th = inp_th.to(self.gpu)
            inp_length = inp_length.to(self.gpu)
            fw_pos = fw_pos.to(self.gpu)
            bw_pos = bw_pos.to(self.gpu)

        batch = Batch(text=inp_th, length=inp_length, fw_pos=fw_pos, bw_pos=bw_pos)

        inp_enb, inp_ex = [], []
        inp_ex_mask = []

        max_sample_length = (self.nb_range + 1) if self.nb_method == 'ngram' else (inp_th.size(0) + 1)
        fw_sample_outputs, bw_sample_outputs = self.lm_model.sample_n('random', batch,
                                                                      max_sample_length=max_sample_length,
                                                                      sample_num=self.sample_num)
        for sample_i in range(self.sample_num):
            fw_sample_seq, bw_sample_seq = fw_sample_outputs[:, sample_i].cpu().numpy(), \
                                           bw_sample_outputs[:, sample_i].cpu().numpy()
            filled_inp = copy.copy(inp)

            len_bw = x_region[0] - nb_region[0]
            len_fw = nb_region[1] - x_region[1]
            if len_bw > 0:
                filled_inp[nb_region[0]:x_region[0]] = lm_id_to_bert_id(bw_sample_seq[-len_bw:], self.tokenizer,
                                                                        self.vocab)
            if len_fw > 0:
                filled_inp[x_region[1] + 1:nb_region[1] + 1] = lm_id_to_bert_id(fw_sample_seq[:len_fw], self.tokenizer,
                                                                                self.vocab)
            inp_enb.append(filled_inp)

            filled_ex, mask_ex = [], []
            flg = False
            for i in range(len(filled_inp)):
                if not x_region[0] <= i <= x_region[1]:
                    filled_ex.append(filled_inp[i])
                    mask_ex.append(inp_mask[i])
                elif not flg:
                    filled_ex.append(self.tokenizer.vocab['[PAD]'])
                    mask_ex.append(inp_mask[i])
                    # flg = True
            filled_ex = np.array(filled_ex, dtype=np.int32)
            mask_ex = np.array(mask_ex, dtype=np.int32)
            inp_ex.append(filled_ex)
            inp_ex_mask.append(mask_ex)

        inp_enb, inp_ex = np.stack(inp_enb), np.stack(inp_ex)
        inp_ex_mask = np.stack(inp_ex_mask)
        inp_enb, inp_ex = torch.from_numpy(inp_enb).long(), torch.from_numpy(inp_ex).long()
        inp_enb_mask, inp_ex_mask = torch.from_numpy(inp_mask).long(), torch.from_numpy(inp_ex_mask).long()

        if self.gpu >= 0:
            inp_enb, inp_ex = inp_enb.to(self.gpu), inp_ex.to(self.gpu)
            inp_enb_mask, inp_ex_mask = inp_enb_mask.to(self.gpu), inp_ex_mask.to(self.gpu)
            segment_ids = segment_ids.to(self.gpu)

        inp_enb_mask = inp_enb_mask.expand(inp_enb.size(0), -1)
        segment_ids = segment_ids.expand(inp_enb.size(0), -1)

        # segment_ids[:, :inp_enb.size(1)]
        # segment_ids[:, :inp_ex.size(1)]
        logits_enb = self.model(input_ids=inp_enb, attention_mask=inp_enb_mask)[0]
        logits_ex = self.model(input_ids=inp_ex, attention_mask=inp_ex_mask)[0]

        contrib_logits = logits_enb - logits_ex  # [B * 2]
        if contrib_logits.size(1) == 2:
            contrib_score = contrib_logits[:, 1] - contrib_logits[:, 0]  # [B]
        else:
            contrib_score = normalize_logit(contrib_logits, label)
        contrib_score = contrib_score.mean()
        return contrib_score.item()

    def explain_single_transformer(self, input_ids, input_mask, segment_ids, region, label=None):
        inp_flatten = input_ids.view(-1).cpu().numpy()
        inp_mask_flatten = input_mask.view(-1).cpu().numpy()

        if self.nb_method == 'ngram':
            mask_regions = self.get_ngram_mask_region(region, inp_flatten)
        else:
            raise NotImplementedError('unknown method %s' % self.nb_method)

        score = self.score(inp_flatten, inp_mask_flatten, segment_ids, [region],
                           mask_regions, label)
        return score

    def score_seq(self, inp_bak, inp_mask, segment_ids, valid_positions, x_regions, nb_regions, lm_entity_span, label):
        x_region = x_regions[0]
        nb_region = nb_regions[0]
        inp = copy.copy(inp_bak)

        inp_length = 0
        for i in range(len(inp_mask)):
            if inp_mask[i] == 1:
                inp_length += 1
            else:
                break

        # mask everything outside the x_region and inside nb region
        inp_lm = copy.copy(inp)
        for i in range(len(inp_lm)):
            if nb_region[0] <= i <= nb_region[1] and not x_region[0] <= i <= x_region[1]:
                inp_lm[i] = self.tokenizer.vocab['[PAD]']

        inp_th = torch.from_numpy(bert_id_to_lm_id(inp_lm[1:inp_length - 1], self.tokenizer, self.vocab)).long().view(
            -1, 1)
        inp_length = torch.LongTensor([inp_length - 2])
        fw_pos = torch.LongTensor([min(x_region[1] + 1 - 1, len(inp) - 2)])
        bw_pos = torch.LongTensor([max(x_region[0] - 1 - 1, -1)])

        if self.gpu >= 0:
            inp_th = inp_th.to(self.gpu)
            inp_length = inp_length.to(self.gpu)
            fw_pos = fw_pos.to(self.gpu)
            bw_pos = bw_pos.to(self.gpu)

        batch = Batch(text=inp_th, length=inp_length, fw_pos=fw_pos, bw_pos=bw_pos)

        inp_enb, inp_ex = [], []
        inp_ex_mask = []

        max_sample_length = (self.nb_range + 1) if self.nb_method == 'ngram' else (inp_th.size(0) + 1)
        fw_sample_outputs, bw_sample_outputs = self.lm_model.sample_n('random', batch,
                                                                      max_sample_length=max_sample_length,
                                                                      sample_num=self.sample_num)
        for sample_i in range(self.sample_num):
            fw_sample_seq, bw_sample_seq = fw_sample_outputs[:, sample_i].cpu().numpy(), \
                                           bw_sample_outputs[:, sample_i].cpu().numpy()
            filled_inp = copy.copy(inp)

            len_bw = x_region[0] - nb_region[0]
            len_fw = nb_region[1] - x_region[1]
            if len_bw > 0:
                filled_inp[nb_region[0]:x_region[0]] = lm_id_to_bert_id(bw_sample_seq[-len_bw:], self.tokenizer,
                                                                        self.vocab)
            if len_fw > 0:
                filled_inp[x_region[1] + 1:nb_region[1] + 1] = lm_id_to_bert_id(fw_sample_seq[:len_fw], self.tokenizer,
                                                                                self.vocab)
            inp_enb.append(filled_inp)

            filled_ex, mask_ex = [], []
            flg = False
            for i in range(len(filled_inp)):
                if not x_region[0] <= i <= x_region[1]:
                    filled_ex.append(filled_inp[i])
                    mask_ex.append(inp_mask[i])
                elif not flg:
                    filled_ex.append(self.tokenizer.vocab['[PAD]'])
                    mask_ex.append(inp_mask[i])
                    # flg = True
            filled_ex = np.array(filled_ex, dtype=np.int32)
            mask_ex = np.array(mask_ex, dtype=np.int32)
            inp_ex.append(filled_ex)
            inp_ex_mask.append(mask_ex)

        inp_enb, inp_ex = np.stack(inp_enb), np.stack(inp_ex)
        inp_ex_mask = np.stack(inp_ex_mask)
        inp_enb, inp_ex = torch.from_numpy(inp_enb).long(), torch.from_numpy(inp_ex).long()
        inp_enb_mask, inp_ex_mask = torch.from_numpy(inp_mask).long(), torch.from_numpy(inp_ex_mask).long()

        if self.gpu >= 0:
            inp_enb, inp_ex = inp_enb.to(self.gpu), inp_ex.to(self.gpu)
            inp_enb_mask, inp_ex_mask = inp_enb_mask.to(self.gpu), inp_ex_mask.to(self.gpu)
            segment_ids = segment_ids.to(self.gpu)

        inp_enb_mask = inp_enb_mask.expand(inp_enb.size(0), -1)
        segment_ids = segment_ids.expand(inp_enb.size(0), -1)
        valid_positions = valid_positions.expand(inp_enb.size(0), -1)

        logits_enb = self.NER_infer_module.getScore(input_ids=inp_enb, input_mask=inp_enb_mask,
                                                    segment_ids=segment_ids[:, :inp_enb.size(1)],
                                                    valid_positions=valid_positions[:, :inp_enb.size(1)],
                                                    lm_entity_span=lm_entity_span,
                                                    gt_label=label)

        logits_ex = self.NER_infer_module.getScore(input_ids=inp_ex, input_mask=inp_ex_mask,
                                                    segment_ids=segment_ids[:, :inp_ex.size(1)],
                                                    valid_positions=valid_positions[:, :inp_ex.size(1)],
                                                    lm_entity_span=lm_entity_span,
                                                    gt_label=label)

        contrib_logits = logits_enb - logits_ex  # Batch size X Entity span length

        score_agg_method = "derive"
        if score_agg_method == "derive":
            # Derive the importance score of the span
            contrib_logits = contrib_logits.mean(axis=1)  # Batch size X 1
            contrib_score = contrib_logits.mean()  # 1 X 1
        elif score_agg_method == "sum":
            # Sum the importance score of the span
            contrib_logits = contrib_logits.mean(axis=0)  # 1 X Entity span length
            contrib_score = contrib_logits.sum()  # 1 X 1
        else:
            raise Exception("Invalid score aggregation method")

        return contrib_score

    def explain_single_transformer_seq(self, input_ids, input_mask, segment_ids, valid_positions, region, lm_entity_span, label=None):
        inp_flatten = input_ids.view(-1).cpu().numpy()
        inp_mask_flatten = input_mask.view(-1).cpu().numpy()

        if self.nb_method == 'ngram':
            mask_regions = self.get_ngram_mask_region(region, inp_flatten)
        else:
            raise NotImplementedError('unknown method %s' % self.nb_method)

        score = self.score_seq(inp_flatten, inp_mask_flatten, segment_ids, valid_positions, [region],
                           mask_regions, lm_entity_span, label)

        return score
