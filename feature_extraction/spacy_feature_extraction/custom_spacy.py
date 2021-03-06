# -*- coding: utf-8 -*-
"""spacy_custom_ner.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1EzfT69yQsw9op6HtUBCh4u7b-bDxbalw
"""

from __future__ import unicode_literals, print_function
import pickle
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse
from spacy.scorer import Scorer

LABEL=['TN', 'PV', 'SD', 'SS', 'SP', 'ES', 'TE', 'PR']

with open ('spacy_train', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

with open ('spacy_test', 'rb') as fp:
    test_data = pickle.load(fp)

def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        annot=annot['entities']
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores


model=None
new_model_name='new_model'
output_dir='content/'
n_iter=20

"""Setting up the pipeline and entity recognizer, and training the new entity."""
if model is not None:
    nlp = spacy.load(model)  # load existing spacy model
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  # create blank Language class
    print("Created blank 'en' model")
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        print(ent[2])
        ner.add_label(ent[2])

if model is None:
    optimizer = nlp.begin_training()
else:
    optimizer = nlp.entity.create_optimizer()

# Get names of other pipes to disable them during training to train only NER
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
        print('Losses', losses)

# Test the trained model
test_text = 'He was struck by H5N1 virus in 2006.'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, ent.text)

# Save model
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.meta['name'] = new_model_name  # rename model
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    # Test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)
    for ent in doc2.ents:
        print(ent.label_, ent.text)


nlp.to_disk("ner_model")

ner_model = spacy.load("ner_model") # for spaCy's pretrained use 'en_core_web_sm'
results = evaluate(ner_model, test_data)

print(results)