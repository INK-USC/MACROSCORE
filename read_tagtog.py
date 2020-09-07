import json, os, xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# tagtog_dir = "./tagtog/plain.html/pool"
tagtog_dir = "./test_annotation/plain.html/pool"
tagtog_annotation = "./test_annotation/ann.parsed_rpp/master/pool"
tagtog_list = os.listdir(tagtog_dir)
tagtog_annotation_list = os.listdir(tagtog_annotation)
print(tagtog_annotation_list)
for file in tagtog_list:
    file_name = file.split('.plain.html')[0]
    print(file_name)
    entities = []
    if file_name+".ann.parsed_rpp" in tagtog_annotation_list:
        with open(tagtog_annotation+"/"+file_name+".ann.parsed_rpp") as json_file:
            data = json.load(json_file)
            for entity in data['entities']:
                entities.append([entity['offsets'][0]['start'], entity['offsets'][0]['start'] +
                                 len(entity['offsets'][0]['text']), entity['offsets'][0]['text'], entity['classId']])


    tree = ET.parse(tagtog_dir+"/"+file)
    root = tree.getroot()
    doc_orig_name = root.get('data-origid')
    doc_text = ' '.join([r.text for r in root[1][0][0][0]])
    for entity in entities:
        start_offset = entity[0]
        end_offset = entity[1]
        assert len(entity[2]) == end_offset-start_offset
        i = 0
        while entity[2] != doc_text[start_offset:end_offset]:
            start_offset+=1
            end_offset+=1

        assert entity[2] == doc_text[start_offset:end_offset]
        entity[0] = start_offset
        entity[1] = end_offset

    print("Text length:", len(doc_text))
    # with open("./tagtog/text/"+doc_orig_name+'.txt', 'w') as f:
    with open("./tagtog/text_RPP/"+doc_orig_name+'.txt', 'w') as f:
        f.write(doc_orig_name + '\n')
        f.write(file + '\n\n')
        f.write(doc_text)

    entity_map = {'e_1':'SS', 'e_2':'ES', 'e_3':'PV', 'e_4':'SD', 'e_5':'TN',
                  'e_6':'TE', 'e_7':'SP', 'e_8':'PR'}

    def takeStart(elem):
        return elem[0]
    entities = sorted(entities, key=takeStart)

    new_doc_text = ''
    prev_offset = 0
    for i in range(len(entities)):
        start_offset = entities[i][0]
        end_offset = entities[i][1]

        new_doc_text = new_doc_text + doc_text[prev_offset:start_offset]
        new_doc_text = new_doc_text + '<' + entity_map[entities[i][3]] + '>'
        new_doc_text = new_doc_text + doc_text[start_offset:end_offset]
        new_doc_text = new_doc_text + '</' + entity_map[entities[i][3]] + '>'

        prev_offset = end_offset


    if len(entities) > 0:
        # os.remove("./tagtog/annotated/" + doc_orig_name + '.txt')
        sents = sent_tokenize(new_doc_text)
        words = [word_tokenize(t) for t in sents]
        print("# sentences:", len(sents))
        print("# words:", len([r for rr in words for r in rr]))
        # with open("./tagtog/tokenized/"+doc_orig_name+'.txt', 'w') as f:
        with open("./tagtog/new_annotated_test/"+doc_orig_name+'.txt', 'w') as f:
            f.write(doc_orig_name + '\n')
            f.write(file + '\n\n')
            for s in words:
                sentence = " ".join(s)
                sentence = re.sub(r'<\s([A-Z][A-Z])\s>', r'<\1>', sentence)
                sentence = re.sub(r'<\s/([A-Z][A-Z])\s>', r'</\1>', sentence)
                #print(sentence)
                f.write(sentence + '\n')

    # if len(entities) == 0:
    #     os.remove("./tagtog/annotated/"+doc_orig_name+'.txt')

