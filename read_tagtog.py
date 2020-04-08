import json, os, xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize

# tagtog_dir = "./tagtog/plain.html/pool"
tagtog_dir = "./tagtog/plain.html_RPP"
tagtog_list = os.listdir(tagtog_dir)
for file in tagtog_list:
    tree = ET.parse(tagtog_dir+"/"+file)
    root = tree.getroot()
    doc_orig_name = root.get('data-origid')
    doc_text = ' '.join([r.text for r in root[1][0][0][0]])
    print("Text length:", len(doc_text))
    # with open("./tagtog/text/"+doc_orig_name+'.txt', 'w') as f:
    with open("./tagtog/text_RPP/"+doc_orig_name+'.txt', 'w') as f:
        f.write(doc_orig_name + '\n')
        f.write(file + '\n\n')
        f.write(doc_text)
    
    sents = sent_tokenize(doc_text)
    words = [word_tokenize(t) for t in sents]
    print("# sentences:", len(sents))
    print("# words:", len([r for rr in words for r in rr]))
    # with open("./tagtog/tokenized/"+doc_orig_name+'.txt', 'w') as f:
    with open("./tagtog/tokenized_RPP/"+doc_orig_name+'.txt', 'w') as f:
        f.write(doc_orig_name + '\n')
        f.write(file + '\n\n')
        for s in words:
            f.write(" ".join(s) + '\n')

