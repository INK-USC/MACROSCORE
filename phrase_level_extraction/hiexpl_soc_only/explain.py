# from algo.soc_lstm import SOCForLSTM
from algo.soc_transformer import SOCForTransformer
import torch
import argparse
from utils.args import get_args
from utils.reader import get_data_iterators_repr
import random, os
from transformers import BertConfig, BertForSequenceClassification
# from nns.model import LSTMMeanRE, LSTMMeanSentiment, LSTMSentiment

def get_args_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    args = parser.parse_args()
    return args

args = get_args()

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    if args.task == 'repr':
        text_field, label_field, train_iter, dev_iter, test_iter, train, dev = \
            get_data_iterators_repr(map_cpu=False)
    else:
        raise ValueError('unknown task')

    iter_map = {'train': train_iter, 'dev': dev_iter, 'test': test_iter}
    if args.task == 'repr':
        tree_path = './data/repr_sentences/trees/%s.csv'
    else:
        raise ValueError

    args.n_embed = len(text_field.vocab)
    args.d_out = 2 if args.task in ['repr'] else len(label_field.vocab)
    args.n_cells = args.n_layers
    args.use_gpu = args.gpu >= 0

    if args.explain_model == 'bert':
        CONFIG_NAME = 'bert_config.json'
        WEIGHTS_NAME = 'pytorch_model.bin'
        output_model_file = os.path.join('%s' % args.resume_snapshot, WEIGHTS_NAME)
        output_config_file = os.path.join('%s' % args.resume_snapshot, CONFIG_NAME)
        # Load a trained model and config that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
        model.load_state_dict(torch.load(output_model_file))
        model.eval()
        if args.gpu >= 0:
            model = model.to(args.gpu)
        if args.method == 'soc':
            lm_model = torch.load(args.lm_path, map_location=lambda storage, location: storage.cuda(args.gpu))
            lm_model.gpu = args.gpu
            lm_model.encoder.gpu = args.gpu
            algo = SOCForTransformer(model, lm_model,
                                     tree_path=tree_path % args.dataset,
                                     output_path='outputs/' + args.task + '/soc_bert_results/soc%s.txt' % args.exp_name,
                                     config=args, vocab=text_field.vocab)
        else:
            raise ValueError('unknown method')
    else:
        raise ValueError('unknown model')
    with torch.no_grad():
        if args.task == 'repr':
            with torch.cuda.device(args.gpu):
                if args.agg:
                    algo.explain_agg('repr')
                else:
                    # algo.explain_token('repr')
                    algo.explain_repr()

