from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from typing import List
import argparse

import flair, torch
flair.device = torch.device('cuda:4')
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
bertmodel = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train flair")
    parser.add_argument("--folder", type=str, help="folder of data")
    parser.add_argument("--idx", type=str, help="chkp idx")
    args = parser.parse_args()
    args = vars(args)
    
    corpus = ColumnCorpus(data_folder='./data_processed', column_format={0: "text", 1: "ner"}, \
                          train_file='train.txt', test_file='test.txt', dev_file='test.txt')
    #corpus.train = [sentence for sentence in corpus.train if len(sentence) > 0]
    #corpus.test = [sentence for sentence in corpus.test if len(sentence) > 0]
    #corpus.dev = [sentence for sentence in corpus.dev if len(sentence) > 0]


    print(corpus.train.total_sentence_count) #27100
    print(corpus.test.total_sentence_count) #8036

    max_len = 512
    for sent in corpus.train:
        if len(sent.tokens) > 128 or len(sent.tokens) <= 1 or sent.tokens[-1].text not in ['.']:
            corpus.train.sentences.remove(sent)
            corpus.train.total_sentence_count -= 1

    for sent in corpus.test:
        if len(sent.tokens) > 128 or len(sent.tokens) <= 1 or sent.tokens[-1].text not in ['.']:
            corpus.test.sentences.remove(sent)
            corpus.test.total_sentence_count -= 1

    print(corpus.train.total_sentence_count) #26916 -> 26869 -> 26548
    print(corpus.test.total_sentence_count) #7833 -> 7827 -> 7755

    tag_type = 'ner'
    
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    scibert_embedding = TransformerWordEmbeddings('allenai/scibert_scivocab_uncased')
    # embedding_types: List[TokenEmbeddings] = [
    #
    #     scibert_embedding,
    #
    #     # comment in this line to use character embeddings
    #     # CharacterEmbeddings(),
    #
    #     # comment in these lines to use flair embeddings
    #     # FlairEmbeddings('news-forward'),
    #     # FlairEmbeddings('news-backward'),
    # ]
    #
    # embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    #
    from flair.models import SequenceTagger
    
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=scibert_embedding,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)
    
    # 6. initialize trainer
    from flair.trainers import ModelTrainer
    
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    
    # 7. start training
    trainer.train('./flair_models/'+args['folder']+'_'+args['idx'],
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=200)
    
    # 8. plot weight traces (optional)
    # from flair.visual.training_curves import Plotter
    # plotter = Plotter()
    # plotter.plot_weights('resources/taggers/example-ner/weights.txt')
