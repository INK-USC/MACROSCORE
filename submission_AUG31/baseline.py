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
        # get scibert
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        self.bertmodel = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.bertmodel.to('cuda')

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
            "model_keywords": ["meta.analysis.significant", "claim-only", "rpp", "best"],
            "model_name": "claim_only_baseline",
            "model_author": "Dong-Ho",
            "model_timestamp": datetime.now().timestamp(),
            "tag": "meta_analysis_significant_classifier_with_claims_only"
        }


    def encode_paper(self, claims):
        embs = []
        for cont in claims:
            input_ids = torch.tensor(self.tokenizer.encode(cont)).unsqueeze(0)  # Batch size 1
            try:
                input_ids = input_ids.to('cuda')
                outputs = self.bertmodel(input_ids)
            except Exception as e:
                print(e)
                continue
            cls = outputs[0][:, 0, :].squeeze().detach().cpu()
            del outputs
            del input_ids
            embs.append(cls)
        final_embs = torch.stack(embs)
        final_embs = final_embs.unsqueeze(0)
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
                for idx, data in enumerate(test_data['data']):
                    claim2 = data['claim2']
                    claim3a = data['claim3a']
                    claim3b = data['claim3b']
                    claim4 = data['claim4']

                    claims = [claim2,claim3a,claim3b,claim4]
                    embs, doc_len = self.encode_paper(claims)
                    if torch.cuda.is_available():
                        embs = embs.cuda()
                    prediction, attn_weight = self.model(embs, 1)

                    print(F.softmax(prediction).cpu().numpy()[0][1])
                    predictions.append(F.softmax(prediction).cpu().numpy()[0][1])

            result = {"result": [{"pid": pid, "score": str(score)} for pid, score in zip(pids, predictions)]}
            self.result.update(result)
            json.dump(self.result, open("../dongho_claims_ta1.json", 'w'))

        elif test == 'covid':
            test_data = json.load(open('../data_processed/covid_scienceparse_classify_data.json', 'r'))
            pids = []
            predictions = []
            with torch.no_grad():
                for idx, data in enumerate(test_data):
                    paper_id = data['paper_id']
                    claim2 = data['claim2']
                    claim3a = data['claim3a']
                    claim3b = data['claim3b']
                    claim4 = data['claim4']

                    claims = [claim2, claim3a, claim3b, claim4]
                    embs, doc_len = self.encode_paper(claims)
                    if torch.cuda.is_available():
                        embs = embs.cuda()
                    prediction, attn_weight = self.model(embs, 1)
                    pids.append(paper_id)

                    print(F.softmax(prediction).cpu().numpy()[0][1])
                    predictions.append(F.softmax(prediction).cpu().numpy()[0][1])

            result = {"result": [{"pid": pid, "score": str(score)} for pid, score in zip(pids, predictions)]}
            self.result.update(result)
            json.dump(self.result, open("../dongho_claims_covid.json", 'w'))

        else:
            test_data = json.load(open('../data_processed/ta3_dry_run.json', 'r'))
            pids = []
            predictions = []
            with torch.no_grad():
                for idx, data in enumerate(test_data['data']):

                    claim2 = data['coded_claim2']
                    claim3a = data['coded_claim3a']
                    claim3b = data['coded_claim3b']
                    claim4 = data['coded_claim4']

                    claims = [claim2, claim3a, claim3b, claim4]
                    embs, doc_len = self.encode_paper(claims)
                    if torch.cuda.is_available():
                        embs = embs.cuda()
                    prediction, attn_weight = self.model(embs, 1)


                    print(F.softmax(prediction).cpu().numpy()[0][1])
                    predictions.append(F.softmax(prediction).cpu().numpy()[0][1])

            result = {"result": [{"pid": pid, "score": str(score)} for pid, score in zip(pids, predictions)]}
            self.result.update(result)
            json.dump(self.result, open("../dongho_claims_covid.json", 'w'))

    def submit_result(self, endpoint="http://ckg03.isi.edu:7789/submission"):
        result = json.load(open('../dongho_scibert_rpp.json', 'r'))
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


