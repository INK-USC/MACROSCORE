import json, os, xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from tqdm import tqdm

if __name__ == "__main__":

    tagtog_base_dir = "./annotations_extra_with_citations"
    tagtog_dir = os.path.join(tagtog_base_dir, "plain.html/pool")
    tagtog_annotation = os.path.join(tagtog_base_dir, "ann.json/master/pool")


    tagtog_list = os.listdir(tagtog_dir)
    tagtog_annotation_list = os.listdir(tagtog_annotation)
    print(tagtog_annotation_list)
    for file in tagtog_list:
        file_name = file.split('.plain.html')[0]
        print(file_name)

        entities = []
        if file_name+".ann.json" in tagtog_annotation_list:
            with open(tagtog_annotation+"/"+file_name+".ann.json") as json_file:
                data = json.load(json_file)
                for entity in data['entities']:
                    entities.append([entity['offsets'][0]['start'], entity['offsets'][0]['start'] +
                                     len(entity['offsets'][0]['text']), entity['offsets'][0]['text'], entity['classId']])


        tree = ET.parse(tagtog_dir+"/"+file)
        root = tree.getroot()
        doc_orig_name = root.get('data-origid')
        doc_text = ' '.join([r.text for r in root[1][0][0][0]])
        print("Text length:", len(doc_text))

        for idx, entity in enumerate(tqdm(entities)):
            start_offset = entity[0]
            end_offset = entity[1]
            assert len(entity[2]) == end_offset-start_offset
            i = 0
            while entity[2] != doc_text[start_offset:end_offset] and end_offset <= len(doc_text):
                start_offset+=1
                end_offset+=1
                # print(start_offset, end_offset)

            if entity[2] == doc_text[start_offset:end_offset]:
                entity[0] = start_offset
                entity[1] = end_offset
            else:
                pass


        out_dir = os.path.join(tagtog_base_dir, "tokenized")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_path = os.path.join(tagtog_base_dir, "tokenized", doc_orig_name+'.txt')
        with open(out_path, 'w') as f:
            f.write(doc_orig_name + '\n')
            f.write(file + '\n\n')
            f.write(doc_text)
        print("Done writing the file {}".format(out_path))

        entity_map_path = os.path.join(tagtog_base_dir, "annotations-legend.json")
        with open(entity_map_path, "r") as f:
            entity_map = json.load(f)

        entities = sorted(entities, key=lambda x: x[0])

        new_doc_text = ''
        prev_offset = 0
        print("Num Entities:", len(entities))
        for i in range(len(entities)):
            start_offset = entities[i][0]
            end_offset = entities[i][1]

            new_doc_text = new_doc_text + doc_text[prev_offset:start_offset]
            new_doc_text = new_doc_text + '<' + entity_map[entities[i][3]] + '>'
            new_doc_text = new_doc_text + doc_text[start_offset:end_offset]
            new_doc_text = new_doc_text + '</' + entity_map[entities[i][3]] + '>'
            prev_offset = end_offset

        if len(entities) > 0:
            sents = sent_tokenize(new_doc_text)
            words = [word_tokenize(t) for t in sents]
            print("# sentences:", len(sents))
            print("# words:", len([r for rr in words for r in rr]))
            out_path = os.path.join(tagtog_base_dir, "tokenized", doc_orig_name + '.txt')
            with open(out_path, 'w') as f:
                f.write(doc_orig_name + '\n')
                f.write(file + '\n\n')
                for s in words:
                    sentence = " ".join(s)
                    sentence = re.sub(r'<\s([A-Z][A-Z])\s>', r'<\1>', sentence)
                    sentence = re.sub(r'<\s/([A-Z][A-Z])\s>', r'</\1>', sentence)
                    f.write(sentence + '\n')

        if len(entities) == 0:
            out_path = os.path.join(tagtog_base_dir, "tokenized", doc_orig_name + '.txt')
            os.remove(out_path)

