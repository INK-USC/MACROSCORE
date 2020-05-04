import numpy as np
from matplotlib import pyplot as plt

multi_word_phrases = open("../AutoPhrase/models/raw_all_autophrase/AutoPhrase_multi-words.txt", 'r').readlines()
multi_word_phrases = [r[:-1].split('\t')[1] for r in multi_word_phrases][:10000]

# treat full TA1 text as a query
TA1_data, curr_data, curr_id = [], [], None
with open("data_processed/raw_all.txt", 'r') as f:
    f.readline()
    curr_id = f.readline()[:-1]
    while True:
        line = f.readline()
        if line in ["", "----NEW DOC----\n"]:
            assert curr_data
            TA1_data.append([curr_id, ' ### '.join(curr_data)])
            curr_data = []
            if line == "":
                break
            curr_id = f.readline()[:-1]
        else:
            curr_data.append(line[:-1])

# calculate avg tf-idf
phrases_tf, phrases_df = np.zeros(len(multi_word_phrases)), np.zeros(len(multi_word_phrases))
for i in range(len(TA1_data)):
    if i % 100 == 0:
        print(i)
    text = TA1_data[i][1]
    for j in range(len(multi_word_phrases)):
        count_i_text = text.count(multi_word_phrases[j])
        if count_i_text:
            phrases_tf[j] += count_i_text
            phrases_df[j] += 1

# remove non-existing phrases
exist_phrases_idx = np.where(phrases_df != 0)[0]
multi_word_phrases = [multi_word_phrases[i] for i in exist_phrases_idx]
phrases_tf = phrases_tf[exist_phrases_idx]
phrases_df = phrases_df[exist_phrases_idx]

phrases_tf = phrases_tf / phrases_tf.max()
phrases_idf = np.log2(len(TA1_data) / phrases_df)
phrases_tfidf = phrases_tf * phrases_idf / len(TA1_data)

# CZI and PMC papers
CZIPMC_data, curr_data, curr_id = [], [], None
with open("data_processed/raw_all_czipmc.txt", 'r') as f:
    f.readline()
    curr_id = f.readline()[:-1]
    while True:
        line = f.readline()
        if line in ["", "----NEW DOC----\n"]:
            assert curr_data
            CZIPMC_data.append([curr_id, ' ### '.join(curr_data)])
            curr_data = []
            if line == "":
                break
            curr_id = f.readline()[:-1]
        else:
            curr_data.append(line[:-1])

# calculate tf-idf for each paper
CZIPMC_phrases_tf, CZIPMC_phrases_df = np.zeros((len(CZIPMC_data), len(multi_word_phrases))), np.zeros(len(multi_word_phrases))
for i in range(len(CZIPMC_data)):
    if i % 100 == 0:
        print(i)
    text = CZIPMC_data[i][1]
    for j in range(len(multi_word_phrases)):
        count_i_text = text.count(multi_word_phrases[j])
        if count_i_text:
            CZIPMC_phrases_tf[i,j] += count_i_text
            CZIPMC_phrases_df[j] += 1

CZIPMC_phrases_tf = CZIPMC_phrases_tf / CZIPMC_phrases_tf.max()
CZIPMC_phrases_df[CZIPMC_phrases_df==0] = np.Inf
CZIPMC_phrases_idf = np.log2(len(CZIPMC_data) / CZIPMC_phrases_df)
CZIPMC_phrases_idf[CZIPMC_phrases_idf==-np.Inf] = 0
CZIPMC_phrases_tfidf = CZIPMC_phrases_tf * CZIPMC_phrases_idf

numerator = (phrases_tfidf * CZIPMC_phrases_tfidf).sum(1)
denominator = np.sqrt((phrases_tfidf**2).sum()) * np.sqrt((CZIPMC_phrases_tfidf**2).sum(1))
denominator[denominator==0] = np.Inf
cosine_sim = numerator / denominator
CZIPMC_files_reordered = sorted(zip([r[0] for r in CZIPMC_data], cosine_sim), key=lambda x:x[1], reverse=True)

# plot hist of cosine sim
plt.hist(cosine_sim, bins=50, color='blue', log=True, alpha=1)
plt.savefig('./data/cosine_scores_dist_cord19_TA1.png', format='png')
plt.clf()


# add new column: mentioned covid19 or not
covid19_mentioned_papers = open("./data/COVID_areas_reordered.csv", 'r').readlines()[1:121]
covid19_mentioned_papers = [r.split(',')[0] for r in covid19_mentioned_papers]
CZIPMC_files_reordered = [list(r) + [True if r[0] in covid19_mentioned_papers else False] for r in CZIPMC_files_reordered]

# reorder the metadata file by cosine similarity
metadata = open("./CORD-19-research-challenge/metadata.csv", 'r').readlines()
headings = metadata[0]
metadata = metadata[1:]
metadata = [r for r in metadata if r.split(',')[0]]
output = ['similarity,mention_covid19,'+headings]
for i in range(len(CZIPMC_files_reordered)):
    if i % 100 == 0:
        print(i)
    curr_sha = CZIPMC_files_reordered[i][0]
    for j in range(len(metadata)):
        if metadata[j].split(',')[0] == curr_sha:
            output.append('%.2f' % CZIPMC_files_reordered[i][1]+','+str(CZIPMC_files_reordered[i][2])+','+metadata[j])
            break

with open('data/CORD19_reordered_with_similarity.csv', 'w') as f:
    for l in output:
        f.write(l)

