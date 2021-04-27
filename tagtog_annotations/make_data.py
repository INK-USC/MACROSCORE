import json
from sklearn.model_selection import train_test_split


def make_data(data, file_path):
    f = open(file_path, 'w')
    for d in data:
        sents = d['sents']
        tags = [['O' for r in rr] for rr in sents]

        for t in d['tags']:
            try:
                tags[t[1][1]][t[1][2]] = 'B-' + t[0]
                for i in range(t[1][2]+1, t[2][2]):
                    tags[t[1][1]][i] = 'I-' + t[0]
            except Exception as e:
                print(e)

        for s, t in zip(sents, tags):
            write = 0
            for t_item in t:
                if 'B-' in t_item:
                    write = 1

            if write == 1:
                f.write('\n'.join([s[i].lower()+'\t'+t[i] for i in range(len(s))]) + '\n\n')
    
    f.close()


data = json.load(open("./data_extra.json", 'r'))

# check if multi-sent spans exist
all_tags = [rr for r in data for rr in r['tags']]
print(len(all_tags))
all_tags = [r for r in all_tags if not r[1][1] != r[2][1]]
print(len(all_tags))

assert all([r[1][1]==r[2][1] for r in all_tags])

# train_dev, test = train_test_split(data, test_size=0.1, random_state=0)
# train, dev = train_test_split(train_dev, test_size=0.1, random_state=0)
train, test = train_test_split(data, test_size=0.2, random_state=0)


make_data(train, "./train.txt")
make_data(test, "./test.txt")
# make_data(data, "./all.txt")
#make_data(data, "./data_processed/test_new.txt")