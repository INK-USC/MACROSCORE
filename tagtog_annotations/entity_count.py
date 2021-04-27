import os

base_dir = "../data_processed/ner_dataset_cleaned/"
train = os.path.join(base_dir, 'train.txt')
test = os.path.join(base_dir, 'test.txt')

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

    sum = 0
    for k,v in label_count.items():
        sum += v

    print("Total = ", sum)
    return label_count

print(read_txt(train))
print(read_txt(test))