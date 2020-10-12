import pandas as pd
from datetime import datetime
import os
import numpy as np
np.random.seed(1337)
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import requests

headers = {
    'Content-type': 'application/pdf',
}

class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embedding_length):
        super(AttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length

        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def attention_net(self, lstm_output, final_state):

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, input, batch_size=None):
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            c_0 = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)
        attn_output, attn_weight = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits, attn_weight


class Model:
    def __init__(self):
        #get scibert
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        self.bertmodel = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.bertmodel.to('cuda')
        # self.save_dir = os.path.join("checkpoint", "save")
        # self.model_name = 'scibert100'

        loadFilename = os.path.join('scibert.tar') #TODO
        if loadFilename:
            checkpoint = torch.load(loadFilename)
            model_checkpoint = checkpoint['model']

        self.model = AttentionModel(1, 2, 100, 768)
        if loadFilename:
            self.model.load_state_dict(model_checkpoint)
            self.model.cuda()

        self.model.eval()

        self.data = []
        self.intermediate_data = []
        self.result = {
            "model_keywords": ["meta.analysis.significant", "scibert", "ta1", "best"],
            "model_name": "scibert_based_predictor",
            "model_author": "Dong-Ho",
            "model_timestamp": datetime.now().timestamp(),
            "tag": "meta_analysis_significant_classifier_with_text_scibert"
        }

    def split_paper(self, content):
        l_total = []
        if len(content.split()) // 50 > 0:
            n = len(content.split()) // 50
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = content.split()[:100]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = content.split()[w * 50:w * 50 + 100]
                l_total.append(" ".join(l_parcial))

        #[contents[0:100],contents[50:150],....]
        return l_total


    def encode_paper(self, content):
        embs = []
        for cont in content:
            split_sentences = self.split_paper(cont['text'])
            section_embs = []
            for sentence in split_sentences:
                input_ids = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
                try:
                    input_ids = input_ids.to('cuda')
                    outputs = self.bertmodel(input_ids)
                except Exception as e:
                    print(e)
                    continue
                cls = outputs[0][:, 0, :].squeeze().detach().cpu()
                del outputs
                del input_ids
                section_embs.append(cls)
            section_final_embs = torch.mean(torch.stack(section_embs), 0)
            embs.append(section_final_embs)
        final_embs = torch.stack(embs)
        final_embs = final_embs.unsqueeze(0)
        #[contents[0:100]-cls, contents[50:150]-cls, ....]
        return final_embs, len(embs)

    # TODO : how it looks like?
    def predict(self, df, parsed_jsons: list = None):
        with torch.no_grad():
            for idx, data in enumerate(test_data):
                paper_id = data['paper_id']
                if 'sections' in parsed_jsons['content']:
                    content = data['content']['sections']
                    embs, doc_len = self.encode_paper(content)
                    if torch.cuda.is_available():
                        embs = embs.cuda()
                    prediction, attn_weight = self.model(embs,1)
                    pids.append(paper_id)
                    print(torch.max(prediction, 1)[1].item())
                    predictions.append(torch.max(prediction, 1)[1].item())
        return result

    def data_loader_example(self):
        meta = pd.read_excel('../data/SCORE_csv.20200526.xlsx')[['pdf_filename', 'paper_id']].to_dict()
        total = 0
        error = 0
        for k in meta['pdf_filename'].keys():
            filename = meta['pdf_filename'][k]
            paper_id = meta['paper_id'][k]

            if os.path.exists('../parsed_ta1/'+filename+'.json'):
                data = json.load(open('../parsed_ta1/'+filename+'.json', 'rb'))
                self.data.append({'paper_id': paper_id, 'filename': filename, 'content': data})
            else:
                print(filename)
                error += 1
            total += 1
        print(total, error)
        json.dump(self.data, open("../data_processed/TA1_scienceparse_classify_data.json", 'w'))

    def get_result(self) -> dict:
        test = ''
        print(test)
        if test == 'rpp':
            test_data = json.load(open('../data_processed/RPP_scienceparse_classify_testdata.json', 'r'))
            pids = []
            predictions = []
            with torch.no_grad():
                for idx, data in enumerate(test_data):
                    paper_id = data['doi']
                    content = data['content']
                    embs, doc_len = self.encode_paper(content)
                    if torch.cuda.is_available():
                        embs = embs.cuda()
                    prediction, attn_weight = self.model(embs, 1)
                    pids.append(paper_id)
                    print(F.softmax(prediction).cpu().numpy()[0][1])
                    predictions.append(F.softmax(prediction).cpu().numpy()[0][1])

            result = {"result": [{"pid": pid, "score": str(score)} for pid, score in zip(pids, predictions)]}
            self.result.update(result)
            json.dump(self.result, open("../dongho_scibert_rpp.json", 'w'))
        elif test == 'ta1':
            test_data = json.load(open('../data_processed/TA1_scienceparse_classify_data.json', 'r'))
            pids = []
            predictions = []
            with torch.no_grad():
                for idx, data in enumerate(test_data):
                    paper_id = data['paper_id']
                    content = data['content']
                    embs, doc_len = self.encode_paper(content)
                    if torch.cuda.is_available():
                        embs = embs.cuda()
                    prediction, attn_weight = self.model(embs, 1)
                    pids.append(paper_id)
                    print(F.softmax(prediction).cpu().numpy()[0][1])
                    predictions.append(F.softmax(prediction).cpu().numpy()[0][1])

            result = {"result": [{"pid": pid, "score": str(score)} for pid, score in zip(pids, predictions)]}
            self.result.update(result)
            json.dump(self.result, open("../dongho_scibert_ta1.json", 'w'))
        elif test == 'covid':
            test_data = json.load(open('../data_processed/covid_scienceparse_classify_data.json', 'r'))
            pids = []
            predictions = []
            with torch.no_grad():
                for idx, data in enumerate(test_data):
                    paper_id = data['paper_id']
                    content = data['content']
                    embs, doc_len = self.encode_paper(content)
                    if torch.cuda.is_available():
                        embs = embs.cuda()
                    prediction, attn_weight = self.model(embs, 1)
                    pids.append(paper_id)

                    print(F.softmax(prediction).cpu().numpy()[0][1])
                    predictions.append(F.softmax(prediction).cpu().numpy()[0][1])

            result = {"result": [{"pid": pid, "score": str(score)} for pid, score in zip(pids, predictions)]}
            self.result.update(result)
            json.dump(self.result, open("../dongho_scibert_covid.json", 'w'))
        else:
            test_data = json.load(open('../data_processed/S.json', 'r'))
            pids = []
            predictions = []
            with torch.no_grad():
                content = test_data['sections']
                embs, doc_len = self.encode_paper(content)
                if torch.cuda.is_available():
                    embs = embs.cuda()

                prediction, attn_weight = self.model(embs, 1)
                values, indices = torch.max(attn_weight, 1)

                splitted = []
                for cont in content:
                    splits = self.split_paper(cont['text'])
                    splitted.extend(splits)

                print(splitted[indices.cpu().item()])

                input_ids = torch.tensor(self.tokenizer.encode(splitted[indices.cpu().item()])).unsqueeze(0)

                try:
                    input_ids = input_ids.to('cuda')
                    outputs = self.bertmodel(input_ids)
                except Exception as e:
                    print(e)

                prediction, attn_weight = self.model(outputs[0], 1)
                sorted, indices = torch.sort(attn_weight)

                print(indices[0][:5])

                for i in indices[0][:5]:
                    print(self.tokenizer.decode(torch.tensor([[input_ids[0][i]]])))

                print(F.softmax(prediction).cpu().numpy()[0][1])
                predictions.append(F.softmax(prediction).cpu().numpy()[0][1])

            result = {"result": [{"pid": pid, "score": str(score)} for pid, score in zip(pids, predictions)]}
            print(result)
            self.result.update(result)


    def submit_result(self, endpoint="http://ckg03.isi.edu:7789/submission"):
        result = json.load(open('../dongho_longformer_covid.json', 'r'))
        response = requests.post(endpoint, json=result)
        print(response)

    def submit_model(self, endpoint="http://ckg03.isi.edu:7789/submission"):
        files = {'file': open(os.path.abspath(__file__), 'r')}
        response = requests.post(endpoint, files=files, json=self.result)
        print(response)

    def retrive_submitted_result(self, endpoint="http://ckg03.isi.edu:7789/submission"):
        # retreive a submission earlier
        response = requests.get(endpoint, json={"tag": "A silly tag", "author": "Ron"})
        # or by default return the latest submission
        response = requests.get(endpoint, json={"author": "Ron"})


model1 = Model()
model1.get_result()
#model1.submit_result()
#model1.data_loader_example()
# # print("load")
# #model1.predict()
# # print("load")
# model1.submit_result()
# # print("load")
# # model1.submit_model()
# #
# # print("load")


