
train = './data_processed/train.txt'
test = './data_processed/test.txt'

def read_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        label_count = {}
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                continue
            if len(line.split()) < 2:
                continue
            word, label = line.split()
            if 'B-' in label:
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1

    return label_count

print(read_txt(train))
print(read_txt(test))