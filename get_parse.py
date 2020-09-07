# import requests
# headers = {
#     'Content-type': 'application/pdf',
# }
# data = open('./pdfs/TA1 papers/Abendroth_AmSocioRev_2014_G8Lr.pdf', 'rb').read()
# response = requests.post('http://ckg03.isi.edu:7788/v1', headers=headers, data=data)
# print(response.parsed_rpp())
#
#
# ta1 = pd.read_csv('data/SCORE_csv.20200526.xlsx')
# #Make whole merge file for TA1
#
# import pandas as pd, os, parsed_rpp
# from nltk.tokenize import word_tokenize

metadata = pd.read_excel('./data/SCORE_csv.20200526.xlsx')[['pdf_filename', 'paper_id']].to_dict()

doi2filename = json.load(open("./data/doi_to_file_name_data.parsed_rpp", 'r'))
doi2filename = {r['doi']:r['file'] for r in doi2filename}

data = []
for k in metadata['Unnamed: 0'].keys():
    paper_id = metadata['Unnamed: 0'][k]
    doi = metadata['DOI'][k]
    O_within_CI_R = metadata['O.within.CI.R'][k]
    Meta_analysis_significant = metadata['Meta.analysis.significant'][k]
    pvalue_label = metadata['pvalue.label'][k]
    Fold_Id = metadata['Fold_Id'][k]
    if not doi in doi2filename:
        print('Warning: DOI "' + doi + '" is not found')
        continue
    filename = doi2filename[doi]
    content = open('./tagtog/tokenized_RPP/'+filename+'.txt', 'r').readlines()[3:]
    data.append({'paper_id':paper_id, 'doi':doi, 'O_within_CI_R':O_within_CI_R, \
                 'Meta_analysis_significant':Meta_analysis_significant, 'pvalue_label':pvalue_label, \
                 'Fold_Id':Fold_Id, 'filename':filename, 'content':content})

json.dump(data, open("./data_processed/TA1_scienceparse_classify_data.parsed_rpp", 'w'))


