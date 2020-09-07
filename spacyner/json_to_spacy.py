
# Convert json file to spaCy format.
import plac
import logging
import argparse
import sys
import os
import json
import pickle

@plac.annotations(input_file=("Input file", "option", "i", str), output_file=("Output file", "option", "o", str))
# # python json_to_spacy.py -i ner.json -o spacy_ner
# def extract_time(json):
#     try:
#         # Also convert to int since update_time will be string.  When comparing
#         # strings, "10" is smaller than "2".
#         return int(json['annotation']['update_time'])
#     except KeyError:
#         return 0
#
# # lines.sort() is more efficient than lines = lines.sorted()
# lines.sort(key=extract_time, reverse=True)

def main(input_file=None, output_file=None):
    try:
        training_data = []
        lines = []
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))
            entities.sort()

            start = 0
            end = 0
            new_label = None
            new_entities = []
            for entity in entities:
                if entity[2].startswith('B-'):
                    if new_label != None:
                        new_entities.append((int(start), int(end), str(new_label)))
                    new_label = entity[2].split('-')[1]
                    start = entity[0]
                    end = entity[1]

                if entity[2].startswith('I-'):
                    end = entity[1]
            if new_label != None:
                new_entities.append((int(start), int(end), str(new_label)))

            training_data.append((text, {"entities": new_entities}))
        print(training_data)
        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

        return training_data

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None


if __name__ == '__main__':
    plac.call(main)

