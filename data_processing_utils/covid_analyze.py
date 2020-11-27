import os, json
from matplotlib import pyplot as plt

metadata = open("./data/COVID_areas.csv", 'r').readlines()
sha = [r.split(",")[0] for r in metadata]
sha = [r for r in sha if r]
all_files = [r+".parsed_rpp" for r in sha]
files_1 = [r for r in all_files if r in os.listdir("./CORD-19-research-challenge/comm_use_subset/comm_use_subset")]
files_2 = [r for r in all_files if r in os.listdir("./CORD-19-research-challenge/custom_license/custom_license")]
files_3 = [r for r in all_files if r in os.listdir("./CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset")]

all_files = ["./CORD-19-research-challenge/comm_use_subset/comm_use_subset/"+r for r in files_1] + \
            ["./CORD-19-research-challenge/custom_license/custom_license/"+r for r in files_2] + \
            ["./CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/"+r for r in files_3]
results = {}
for f in all_files:
    content = json.load(open(f, 'r'))
    mention_count = 0
    text = ' '.join([content['metadata']['title'].lower()] + [r['text'].lower() for r in content['body_text']] + \
                    [r['text'].lower() for r in content['abstract']])
    mention_count += text.count("covid-19") + text.count("covid19") + text.count("covid 19") + \
                     text.count("2019-ncov") + text.count("2019ncov") + text.count("2019 ncov") + \
                     text.count("2019 novel coronavirus") + text.count("2019-novel-coronavirus") + \
                     text.count("sars-cov-2") + text.count("sars cov 2") + text.count("sarscov2") + \
                     text.count("2019 novel coronavirus") + text.count("2019-novel-coronavirus") + \
                     text.count("novel coronavirus 2019") + text.count("novel-coronavirus-2019") + \
                     text.count("novel coronavirus pneumonia") + text.count("novel-coronavirus-pneumonia") + \
                     text.count("severe acute respiratory syndrome coronavirus 2") + \
                     text.count("severe-acute-respiratory-syndrome-coronavirus-2")
    
    results[f] = mention_count

plt.hist(list(results.values()), range=[1,59], bins=59)
plt.savefig('./plots/covid19_mention_dist.png', format='png')
plt.clf()

mentioned_files = [r[0] for r in results.items() if r[1]]
with open("./plots/covid19_papers.txt", 'w') as fout:
    for f in mentioned_files:
        content = json.load(open(f, 'r'))
        title = content['metadata']['title']
        if title:
            fout.write(title+'\n')

# reorder the csv file for Daniel
sha = [r.split(",")[0] for r in metadata]
mentioned_files_sha = [r.split('/')[-1][:-5] for r in mentioned_files]
mentioned_files_pos = [sha.index(r) for r in mentioned_files_sha]
new_order = [0] + mentioned_files_pos + [i for i in range(len(sha)) if not i in [0] + mentioned_files_pos]
new_metadata = [metadata[i] for i in new_order]
# open("./data/COVID_areas_reordered.csv", 'w').write(''.join(new_metadata))

# keep only papers with full text available
kept_metadata = [r for r in new_metadata if r[0]!=',']
open("./data/COVID_areas_reordered.csv", 'w').write(''.join(kept_metadata))