train = open("train.txt", 'r').readlines()
test = open("test_new.txt", 'r').readlines()

def conll_format(data):
    sents, labels, curr_sent, curr_label = [], [], [], []
    for line in data:
        line = line[:-1]
        if not line:
            sents.append(curr_sent)
            labels.append(curr_label)
            curr_sent, curr_label = [], []
        else:
            w, l = line.split('\t')
            curr_sent.append(w)
            curr_label.append(l)

    sents = '\n'.join([' '.join(s) for s in sents]) + '\n'
    labels = '\n'.join([' '.join(l) for l in labels]) + '\n'
    return sents, labels

sents, labels = conll_format(train)
open("eng.train.src", 'w').write(sents)
open("eng.train.trg", 'w').write(labels)

sents, labels = conll_format(test)
open("eng.test_new.src", 'w').write(sents)
open("eng.test_new.trg", 'w').write(labels)