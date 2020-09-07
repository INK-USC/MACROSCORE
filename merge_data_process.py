# import excel2json
# json_file = excel2json.convert_from_file('data/new_data.xlsx')
# print(json_file)

# import json, os
# import pandas as pd
#
# scienceparse_dir = "./parsed_rpp"
# scienceparse_list = os.listdir(scienceparse_dir)
# scienceparse_list = sorted(scienceparse_list)
#
# prev_data = pd.read_excel('./data/RPPdata.xlsx')[['DOI']].to_dict()
# print(prev_data)
# new_data = []
# for k in prev_data['DOI'].keys():
#     page_data = {}
#     page_data['doi'] = prev_data['DOI'][k]
#     if str(page_data['doi']).replace('/','_') + '.json' in scienceparse_list:
#         with open(scienceparse_dir+"/"+str(page_data['doi']).replace('/','_') + '.json', 'r') as scienceparse_file:
#             json_file = json.load(scienceparse_file)
#             if 'sections' in json_file:
#                 page_data['content'] = json_file['sections']
#                 new_data.append(page_data)
#             else:
#                 print(page_data['doi'])
#
# with open('./data_processed/RPP_scienceparse_classify_testdata.json', 'w') as outfile:
#     json.dump(new_data, outfile)


#
import json, os
import pandas as pd

scienceparse_dir = "./parsed_ta3_covid"
scienceparse_list = os.listdir(scienceparse_dir)
scienceparse_list = sorted(scienceparse_list)

metadata = pd.read_excel('./data/covid_ta3.xlsx')[['ta3_pid','coded_claim2','coded_claim3a','coded_claim3b','coded_claim4','pdf_filename']].to_dict()

new_data = []
for k in metadata['ta3_pid'].keys():
    page_data = {}
    page_data['paper_id'] = metadata['ta3_pid'][k]
    page_data['claim2'] = metadata['coded_claim2'][k]
    page_data['claim3a'] = metadata['coded_claim3a'][k]
    page_data['claim3b'] = metadata['coded_claim3b'][k]
    page_data['claim4'] = metadata['coded_claim4'][k]
    pdffilename = metadata['pdf_filename'][k]
    if str(pdffilename) + '.json' in scienceparse_list:
        with open(scienceparse_dir+"/"+str(pdffilename) + '.json', 'r') as scienceparse_file:
            json_file = json.load(scienceparse_file)
        if 'sections' in json_file:
            page_data['content'] = json_file['sections']
        else:
            print(pdffilename)
    new_data.append(page_data)
with open('./data_processed/covid_scienceparse_classify_data.json', 'w') as outfile:
    json.dump(new_data, outfile)


#
# import os
# #os.rename(r'file path\OLD file name.file type',r'file path\NEW file name.file type')
# import json, os
# import pandas as pd
#
# scienceparse_dir = "./parsed_ta1"
# scienceparse_list = os.listdir(scienceparse_dir)
# scienceparse_list = sorted(scienceparse_list)
#
# metadata = pd.read_excel('./data/SCORE_csv.20200526.xlsx')[['paper_id','coded_claim2','coded_claim3a','coded_claim3b','coded_claim4','pdf_filename']].to_dict()
# new_data = []
# count=0
# for k in metadata['pdf_filename'].keys():
#     page_data = {}
#     pdffilename = metadata['pdf_filename'][k]
#     page_data['paper_id'] = metadata['paper_id'][k]
#     page_data['claim2'] = metadata['coded_claim2'][k]
#     page_data['claim3a'] = metadata['coded_claim3a'][k]
#     page_data['claim3b'] = metadata['coded_claim3b'][k]
#     page_data['claim4'] = metadata['coded_claim4'][k]
#     if str(pdffilename) + '.json' in scienceparse_list:
#         with open(scienceparse_dir+"/"+str(pdffilename) + '.json', 'r') as scienceparse_file:
#             json_file = json.load(scienceparse_file)
#         if 'sections' in json_file:
#             page_data['content'] = json_file['sections']
#             new_data.append(page_data)
#         else:
#             print(pdffilename)
#             count += 1
#     else:
#         print(pdffilename)
#         count += 1
# print(count)
#
# with open('./data_processed/TA1_scienceparse_classify_data.json', 'w') as outfile:
#     json.dump(new_data, outfile)
