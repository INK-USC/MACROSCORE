import os, re, json
from collections import defaultdict, Counter


keep_variables = ['SS', 'TN', 'TE', 'SD', 'ES', 'PV']


annotated_dir = "./tagtog/annotated"
annotated_files = os.listdir(annotated_dir)
annotated_files = [r for r in annotated_files if r != "0_Labels.txt"]

all_labels = open(annotated_dir+'/0_Labels.txt').readlines()[1:]
all_labels = [r.split(" # ")[0] for r in all_labels]
all_labels = [r.split(",")[1] for r in all_labels]
labels = [r for r in all_labels if r in keep_variables]
all_tags = ['<'+l+'>' for l in all_labels] + ['</'+l+'>' for l in all_labels]
tags = ['<'+l+'>' for l in labels] + ['</'+l+'>' for l in labels]

data = []
for f in annotated_files:
    print(f)
    content = open(annotated_dir+'/'+f).readlines()
    content = [r[:-1] for r in content]
    # if content[2] == "<EX>1</EX>":
        # curr_data["EX"] = 1
    # elif content[2] == "<EX>0</EX>":
        # curr_data["EX"] = 0
    # else:
        # print("EX: Incorrect annotation")
        # assert False
    
    # content starts from line #4
    content = content[3:]
    
    # validate annotation
    content = [r.split(" ") for r in content]
    content_tags = [r for rr in content for r in rr if re.match("</?..>", r)]
    assert all([r in all_tags for r in content_tags])
    unclosed_tags = defaultdict(lambda: 0)
    for t in content_tags:
        if t[1] != "/":
            unclosed_tags[t[1:3]] += 1
        else:
            unclosed_tags[t[2:4]] -= 1
    
    if 1 in unclosed_tags.values():
        print("Annotation Error:")
        print(dict(unclosed_tags))
        assert False
    
    # process data
    # tag: [tag_name, start_position, end_position]
    # position: [doc_word_idx, sent_idx, sent_word_idx]
    doc_word_counter, sent_counter, sent_word_counter = 0, 0, 0
    sents_notags = []
    unclosed_tags, content_tags = {}, []
    for words in content:
        sent_word_counter = 0
        words_notags = []
        for word in words:
            if re.match("</?..>", word):
                if word[1] != "/":
                    unclosed_tags[word[1:3]] = [doc_word_counter, sent_counter, sent_word_counter]
                else:
                    content_tags.append([word[2:4], unclosed_tags[word[2:4]], [doc_word_counter, sent_counter, sent_word_counter]])
            else:
                words_notags.append(word)
                doc_word_counter += 1
                sent_word_counter += 1
        
        sents_notags.append(words_notags)
        sent_counter += 1
    
    content_tags = [r for r in content_tags if r[0] in labels]
    print("Label statistics:")
    print(dict(Counter([r[0] for r in content_tags])))
    print()
    
    curr_data = {"file": f, "sents": sents_notags, "tags": content_tags}
    data.append(curr_data)

json.dump(data, open("./data_processed/data.json", 'w'))