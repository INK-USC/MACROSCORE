import argparse
import flair, torch
from transformers import BertTokenizer, BertModel
import os

from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the flair model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to store the model checkpoint")
    parser.add_argument("--gpu_device", type=str, default="cuda:0", help="GPU device number to use")
    parser.add_argument("--bert_model_name", type=str, default="allenai/scibert_scivocab_uncased", help="Bert model name")

    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_num_tokens", type=int, default=128, help="Maximum number of tokens to consider for each sentence")

    args = parser.parse_known_args()[0]
    print(args)


    flair.device = torch.device(args.gpu_device)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name, do_lower_case=True)
    bertmodel = BertModel.from_pretrained(args.bert_model_name)
    
    corpus = ColumnCorpus(data_folder=args.data_dir, column_format={0: "text", 1: "ner"},
                          train_file='trainplustest.txt', dev_file='test_new.txt', test_file='test_new.txt')

    print(len(corpus.train), len(corpus.dev), len(corpus.test))

    # Filter the dataset to remove the sentences that are longer than the max sequence length:
    filtered_sent = []
    for sent in corpus.train:
        if len(sent.tokens) <= args.max_num_tokens and len(sent.tokens) > 1 and sent.tokens[-1].text in ['.']:
            filtered_sent.append(sent)
    corpus._train = filtered_sent

    filtered_sent = []
    for sent in corpus.dev:
        if len(sent.tokens) <= args.max_num_tokens and len(sent.tokens) > 1 and sent.tokens[-1].text in ['.']:
            filtered_sent.append(sent)
    corpus._dev = filtered_sent

    filtered_sent = []
    for sent in corpus.test:
        if len(sent.tokens) <= args.max_num_tokens and len(sent.tokens) > 1 and sent.tokens[-1].text in ['.']:
            filtered_sent.append(sent)
    corpus._test= filtered_sent

    print(len(corpus.train), len(corpus.dev), len(corpus.test))

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    scibert_embedding = TransformerWordEmbeddings(args.bert_model_name)


    # Create the sequence tagger model:
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=scibert_embedding,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # Initialize the trainer and start the training:
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    output_path = os.path.join(args.output_dir, "scibert_ner_model")
    trainer.train(output_path,
                  learning_rate=args.learning_rate,
                  mini_batch_size=args.batch_size,
                  max_epochs=args.num_epochs)
    
    # Plot the weight curves:
    # plotter = Plotter()
    # plotter.plot_weights('resources/taggers/example-ner/weights.txt')
