

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train flair")
    parser.add_argument("--folder", type=str, help="folder of data")
    parser.add_argument("--idx", type=str, help="chkp idx")
    args = parser.parse_args()
    args = vars(args)
    
    corpus = ColumnCorpus(data_folder='./data_processed', column_format={0: "text", 1: "ner"}, \
                          train_file='train.txt', test_file='test.txt', dev_file='test.txt')
import spacy
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer

nlp = spacy.load('en', entity=False, parser=False)

doc_list = []
doc = nlp('Llamas make great pets.')
doc_list.append(doc)
gold_list = []
gold_list.append(GoldParse(doc, [u'ANIMAL', u'O', u'O', u'O']))

ner = EntityRecognizer(nlp.vocab, entity_types=['ANIMAL'])
ner.update(doc_list, gold_list)


import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer

def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

# example run

examples = [
    ('Who is Shaka Khan?',
     [(7, 17, 'PERSON')]),
    ('I like London and Berlin.',
     [(7, 13, 'LOC'), (18, 24, 'LOC')])
]

ner_model = spacy.load(ner_model_path) # for spaCy's pretrained use 'en_core_web_sm'
results = evaluate(ner_model, examples)