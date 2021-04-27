import torchtext as tt
from nltk import Tree
import pickle, random
import torch
from utils.args import get_args, makedirs
import os
from transformers import BertTokenizer
import csv, json
import nltk
import re

args = get_args()

def save_vocab(path, vocab):
    f = open(path, 'wb')
    pickle.dump(vocab, f)
    f.close()

def load_vocab(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    return obj

def handle_vocab(vocab_path, field, datasets, vector_cache='', train_lm=False, max_size=None):
    create_vocab = False
    if os.path.isfile(vocab_path):
        print('loading vocab from %s' % vocab_path)
        vocab = load_vocab(vocab_path)
        field.vocab = vocab
    else:
        print('creating vocab')
        makedirs('vocab')
        field.build_vocab(*datasets, max_size=max_size)
        vocab = field.vocab
        if '<s>' not in field.vocab.stoi:
            field.vocab.itos.append('<s>')
            field.vocab.stoi['<s>'] = len(field.vocab.itos) - 1
        if '</s>' not in field.vocab.stoi:
            field.vocab.itos.append('</s>')
            field.vocab.stoi['</s>'] = len(field.vocab.itos) - 1
        save_vocab(vocab_path, vocab)
        create_vocab = True

    if vector_cache != '' and not vector_cache.startswith('none'):
        if args.word_vectors or create_vocab:
            if os.path.isfile(vector_cache):
                field.vocab.vectors = torch.load(vector_cache)
            else:
                field.vocab.load_vectors(args.word_vectors)
                for i in range(field.vocab.vectors.size(0)):
                    if field.vocab.vectors[i,0].item() == 0 and field.vocab.vectors[i,1].item() == 0:
                        field.vocab.vectors[i].uniform_(-1,1)
                makedirs(os.path.dirname(vector_cache))
                torch.save(field.vocab.vectors, vector_cache)

    if train_lm:
        v = torch.zeros(2, field.vocab.vectors.size(-1))
        field.vocab.vectors = torch.cat([field.vocab.vectors, v], 0)

def compute_mapping(tokens, bert_tokens):
    mapping = []
    i, j = 0, 0
    while i < len(tokens):
        t = ''
        while len(t) < len(tokens[i]):
            try:
                t += str(bert_tokens[j]).replace('##','')
                j += 1
            except:
                break
        if len(t) > len(tokens[i]):
            print('warning: mapping mismatch')
            break
        mapping.append(j)
        i += 1

    return mapping

def convert_to_bert_tokenization(tokens, bert_tokenizer, return_mapping=False):
    text = ' '.join(tokens)
    bert_tokens = bert_tokenizer.tokenize(text)
    # compute mapping
    if return_mapping:
        mapping = compute_mapping(tokens, bert_tokens)
        return bert_tokens, mapping
    else:
        return bert_tokens

def cleanText(input_text):
    # input_text = re.sub(r'\W+', '', input_text)
    input_text = input_text.replace("...", "")
    return input_text

def get_parsed_example_repr_ner_from_tree_string(tree_string, bert_tokenizer):
    tree = Tree.fromstring(tree_string)
    tokens = tree.leaves()

    if bert_tokenizer is not None:
        tokens, mapping = convert_to_bert_tokenization(tokens, bert_tokenizer, return_mapping=True)

    example = tt.data.Example()
    example.text = tokens
    example.length = len(tokens)
    example.offset = None
    example.label = None

    if bert_tokenizer is not None:
        example.mapping = mapping

    return example

def get_parsed_example_repr_ner_from_text(text, bert_tokenizer):
    tokens = list(text.split())

    if bert_tokenizer is not None:
        tokens, mapping = convert_to_bert_tokenization(tokens, bert_tokenizer, return_mapping=True)

    example = tt.data.Example()
    example.text = tokens
    example.length = len(tokens)
    example.offset = None
    example.label = None

    if bert_tokenizer is not None:
        example.mapping = mapping

    return example

def get_examples_repr_from_trees(path, train_lm, bert_tokenizer=None):
    f = open(path)
    header = f.readline()
    reader = csv.reader(f)
    examples = []
    for i, line in enumerate(reader):

        tree = Tree.fromstring(line[-1].lower())
        tokens = tree.leaves()
        label = int(float(line[1]))

        if bert_tokenizer is not None:
            tokens, mapping = convert_to_bert_tokenization(tokens[:100], bert_tokenizer, return_mapping=True)

        if train_lm:
            tokens = ['<s>'] + tokens[:50] + ['</s>']

        if args.filter_length_gt != -1 and len(tokens) >= args.filter_length_gt:
            continue

        example = tt.data.Example()
        example.text = tokens
        example.length = len(tokens)
        example.offset = i
        example.label = label

        if bert_tokenizer is not None:
            example.mapping = mapping

        examples.append(example)
        if args.explain_model == 'bert':
            if i == args.stop: break

    return examples

def get_examples_repr(path, train_lm, bert_tokenizer=None):
    f = open(path)
    header = f.readline()
    reader = csv.reader(f)
    examples = []
    for i, line in enumerate(reader):
        tokens = nltk.word_tokenize(cleanText(line[1]))
        label = int(float(line[2]))

        if bert_tokenizer is not None:
            tokens, mapping = convert_to_bert_tokenization(tokens[:100], bert_tokenizer, return_mapping=True)

        if train_lm:
            tokens = ['<s>'] + tokens[:50] + ['</s>']

        if args.filter_length_gt != -1 and len(tokens) >= args.filter_length_gt:
            continue

        example = tt.data.Example()
        example.text = tokens
        example.length = len(tokens)
        example.offset = i
        example.label = label

        if bert_tokenizer is not None:
            example.mapping = mapping

        examples.append(example)
        if args.explain_model == 'bert':
            if i == args.stop: break

    return examples

def get_data_iterators_repr(train_lm=False, map_cpu=False):
    text_field = tt.data.Field(lower=args.lower)
    label_field = tt.data.LabelField(sequential=False, unk_token=None)
    length_field = tt.data.Field(sequential=False, use_vocab=False)
    offset_field = tt.data.Field(sequential=False, use_vocab=False)

    data_path = "./data/repr_claims/repr_claims_data.csv"
    bert_tokenizer = None
    if args.use_bert_tokenizer:
        bert_tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True, cache_dir='scibert/cache')

    examples = get_examples_repr(data_path, train_lm, bert_tokenizer=bert_tokenizer)
    train_size = int(0.8 * len(examples))
    train_examples = examples[:train_size]
    dev_examples = test_examples = examples[train_size:]

    train, dev, test = (tt.data.Dataset(ex, [('text', text_field), ('length', length_field), ('offset', offset_field), ('label',label_field)])
                        for ex in [train_examples, dev_examples, test_examples])

    vocab_path = 'vocab/vocab_repr_claims.pkl' if not args.use_bert_tokenizer else 'vocab/vocab_repr_claims_bert.pkl'
    if args.fix_test_vocab and not args.use_bert_tokenizer:
        vocab_path = 'vocab/vocab_repr_claims_fix.pkl'

    c_postfix = '.repr'
    if args.use_bert_tokenizer:
        c_postfix += '.bert'
    if args.fix_test_vocab:
        c_postfix += '.fix'
    handle_vocab(vocab_path, text_field, (train, test), args.vector_cache + c_postfix, train_lm, max_size=20000)
    label_field.build_vocab(train)
    train_iter, dev_iter, test_iter = (
        tt.data.BucketIterator(x, batch_size=args.batch_size, device=args.gpu if not map_cpu else 'cpu', shuffle=False)
        for x in (train, dev, test))
    return text_field, label_field, train_iter, dev_iter, test_iter, train, dev

# def get_data_iterators_repr(train_lm=False, map_cpu=False):
#     text_field = tt.data.Field(lower=args.lower)
#     label_field = tt.data.LabelField(sequential=False, unk_token=None)
#     length_field = tt.data.Field(sequential=False, use_vocab=False)
#     offset_field = tt.data.Field(sequential=False, use_vocab=False)
#
#     data_path = "./data/repr/repr_data.csv"
#     bert_tokenizer = None
#     if args.use_bert_tokenizer:
#         bert_tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True, cache_dir='scibert/cache')
#
#     examples = get_examples_repr(data_path, train_lm, bert_tokenizer=bert_tokenizer)
#     train_size = int(0.8 * len(examples))
#     train_examples = examples[:train_size]
#     dev_examples = test_examples = examples[train_size:]
#
#     train, dev, test = (tt.data.Dataset(ex, [('text', text_field), ('length', length_field), ('offset', offset_field), ('label',label_field)])
#                         for ex in [train_examples, dev_examples, test_examples])
#
#     vocab_path = 'vocab/vocab_repr.pkl' if not args.use_bert_tokenizer else 'vocab/vocab_repr_bert.pkl'
#     if args.fix_test_vocab and not args.use_bert_tokenizer:
#         vocab_path = 'vocab/vocab_repr_fix.pkl'
#
#     c_postfix = '.repr'
#     if args.use_bert_tokenizer:
#         c_postfix += '.bert'
#     if args.fix_test_vocab:
#         c_postfix += '.fix'
#     handle_vocab(vocab_path, text_field, (train, test), args.vector_cache + c_postfix, train_lm, max_size=20000)
#     label_field.build_vocab(train)
#     train_iter, dev_iter, test_iter = (
#         tt.data.BucketIterator(x, batch_size=args.batch_size, device=args.gpu if not map_cpu else 'cpu', shuffle=False)
#         for x in (train, dev, test))
#     return text_field, label_field, train_iter, dev_iter, test_iter, train, dev

# For LSTM Language model training using the NER-TC dataset
def get_examples_repr_ner(path, train_lm, bert_tokenizer=None):
    with open(path, "r") as f:
        data_lines = f.readlines()
    examples = []
    for idx, cur_line in enumerate(data_lines):
        cur_line = cur_line.strip()
        cur_dict = json.loads(cur_line)
        tokens = list(cur_dict["sentence"].split())
        label = int(cur_dict["label_idx"])

        if bert_tokenizer is not None:
            tokens, mapping = convert_to_bert_tokenization(tokens[:150], bert_tokenizer, return_mapping=True)

        if train_lm:
            tokens = ['<s>'] + tokens[:50] + ['</s>']

        if args.filter_length_gt != -1 and len(tokens) >= args.filter_length_gt:
            continue

        example = tt.data.Example()
        example.text = tokens
        example.length = len(tokens)
        example.offset = idx
        example.label = label

        if bert_tokenizer is not None:
            example.mapping = mapping

        examples.append(example)

    return examples

def get_data_iterators_repr_ner(train_lm=False, map_cpu=False):
    text_field = tt.data.Field(lower=args.lower)
    label_field = tt.data.LabelField(sequential=False, unk_token=None)
    length_field = tt.data.Field(sequential=False, use_vocab=False)
    offset_field = tt.data.Field(sequential=False, use_vocab=False)

    data_path = "./data/ner_dataset_tc/{}.jsonl"
    bert_tokenizer = None
    if args.use_bert_tokenizer:
        bert_tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir='scibert/cache')

    train_examples = get_examples_repr_ner(data_path.format("train"), train_lm, bert_tokenizer=bert_tokenizer)
    dev_examples = test_examples = get_examples_repr_ner(data_path.format("test"), train_lm, bert_tokenizer=bert_tokenizer)

    train, dev, test = (tt.data.Dataset(ex, [('text', text_field), ('length', length_field), ('offset', offset_field), ('label',label_field)])
                        for ex in [train_examples, dev_examples, test_examples])

    vocab_path = 'vocab/vocab_repr_ner.pkl' if not args.use_bert_tokenizer else 'vocab/vocab_repr_ner_bert.pkl'
    if args.fix_test_vocab and not args.use_bert_tokenizer:
        vocab_path = 'vocab/vocab_repr_ner_fix.pkl'

    c_postfix = '.repr'
    if args.use_bert_tokenizer:
        c_postfix += '.bert'
    if args.fix_test_vocab:
        c_postfix += '.fix'
    handle_vocab(vocab_path, text_field, (train, test), args.vector_cache + c_postfix, train_lm, max_size=20000)
    label_field.build_vocab(train)
    train_iter, dev_iter, test_iter = (
        tt.data.BucketIterator(x, batch_size=args.batch_size, device=args.gpu if not map_cpu else 'cpu', shuffle=False)
        for x in (train, dev, test))
    return text_field, label_field, train_iter, dev_iter, test_iter, train, dev