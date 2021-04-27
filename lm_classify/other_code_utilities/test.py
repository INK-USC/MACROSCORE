
import pandas as pd, os
import random, sys

psycho_features = pd.read_excel('../data/RPP_psycho_correlation.xlsx').to_dict()


psycho_categories = []
i = 0
for k in psycho_features['feature'].keys():
    if i == 50:
        break
    category = psycho_features['feature'][k]
    psycho_categories.append(category)
    i+=1

import csv
csv.field_size_limit(sys.maxsize)
tsv_file = open('../data/plf_lexica.tsv')
read_tsv = csv.reader(tsv_file, delimiter='\t', quoting=csv.QUOTE_NONE)

count = 0


column_dict = dict()
row_dict = dict()
other_rows = []
for row in read_tsv:
    if count==0:
        for i, x in enumerate(row):
            column_dict[x] = i
    else:
        #print(len(row))
        if len(row) == 327:
            other_rows.append(row)
    count+=1

final_dict = dict()
for category in psycho_categories:
    final_dict[category] = []
    index = column_dict[category]
    if category.split('-')[0] == 'M':
        for row in other_rows:
            if row[index+1] != '0.0':
                final_dict[category].append(row[0])
    if category.split('-')[0] == 'E':
        for row in other_rows:
            if abs(float(row[index+1])) > 2.5:
                final_dict[category].append(row[0])
    if category.split('-')[0] == 'L':
        for row in other_rows:
            if abs(float(row[index + 1])) == 1:
                final_dict[category].append(row[0])
        # else:
        #     print(row[inde])


print(final_dict)
for key in final_dict:
    print(key)
    print(len(final_dict[key]))


import pickle

f = open("file.pkl","wb")
pickle.dump(final_dict,f)
f.close()