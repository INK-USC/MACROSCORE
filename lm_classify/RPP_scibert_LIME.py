import numpy as np
np.random.seed(1337)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from sklearn.metrics import *
import os
import lime.lime_text
from transformers import BertTokenizer, BertModel

def split_paper(content):
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

    return l_total

class SimpleClassification(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassification, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x, batch_size):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out, batch_size


save_dir = os.path.join("checkpoint", "save")
model_name = 'sentence'
loadFilename = os.path.join(save_dir, model_name, '9_checkpoint.tar')
if loadFilename:
    checkpoint = torch.load(loadFilename)
    model_checkpoint = checkpoint['model']
    opt_checkpoint = checkpoint['opt']

sentence_model = SimpleClassification(768, 100, 2)
if loadFilename:
    sentence_model.load_state_dict(model_checkpoint)
    sentence_model.cuda()

class Prediction:

    def __init__(self, seq_length):

        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')

        self.max_seq_length = seq_length
        self.model.to("cuda")
        self.device = "cuda"

    def predict_label(self, text):
        input_ids, input_mask, segment_ids = self.convert_text_to_features(text)
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids, input_mask)

        logits = outputs[0]
        logits = F.softmax(logits, dim=1)
        # print(logits)
        logits_label = torch.argmax(logits, dim=1)
        label = logits_label.detach().cpu().numpy()

        # print("logits label ", logits_label)
        logits_confidence = logits[0][logits_label]
        label_confidence_ = logits_confidence.detach().cpu().numpy()
        # print("logits confidence ", label_confidence_)

        return label, label_confidence_


    def _truncate_seq_pair(self, tokens_a, max_length):
        while True:
            total_length = len(tokens_a)
            if total_length <= max_length:
                break
            if len(tokens_a) > max_length:
                tokens_a.pop()

    def convert_text_to_features(self, text):
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        sequence_a_segment_id = 0
        cls_token_segment_id = 1
        pad_token_segment_id = 0
        mask_padding_with_zero = True
        pad_token = 0
        tokens_a = self.tokenizer.tokenize(text)

        self._truncate_seq_pair(tokens_a, self.max_seq_length - 2)

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)

        return input_ids, input_mask, segment_ids

    def predictor(self, text):

        examples = []

        for example in text:
            examples.append(self.convert_text_to_features(example))

        results = []
        for example in examples:
            with torch.no_grad():
                outputs = self.model(example[0], example[1], example[2])
            cls = outputs[0][:, 0, :].squeeze()
            prediction, _ = sentence_model(cls, 1)
            logits = F.softmax(prediction, dim = 0)
            results.append(logits.cpu().detach().numpy())

        results_array = np.array(results)
        print(results_array.shape)
        return results_array


prediction = Prediction(seq_length = 200)
label_names = ["not meta significant", "meta significant"]
explainer = lime.lime_text.LimeTextExplainer(class_names=label_names)
s_json = 'suggested that older adults were more likely to ignore social distancing guidelines. Research Design and Methods: We conducted two online studies to examine temporal discounting of monetary, health, and social rewards, COVID-19 beliefs, social distancing behaviors, and mental health symptoms. We used the initial study (N = 233) to form our hypotheses about social distancing behaviors and we ran the second, pre-registered study (N = 243) to determine if these relationships replicated. Results: We found that although older adults were more likely to prefer smaller, sooner (i.e., temporal discount) social and health-related rewards in decision-making tasks, there were no adult'
p_json = 'positions in and the nature of the work they conduct on behalf of their organizations.1 Accordingly, variation in individuals’ perceptions of red tape may be related not just to the personal and perceptual factors emphasized in prior research and theory, but also to the variable nature of work and the different organizational environments to which individuals are exposed as they develop and progress in their careers.2 Given that employees’ perceptions of organizational rules and procedures have real and direct consequences on individual performance (DeHart-Davis and Pandey 2005; Scott and Pan- dey 2000), it is important to address factors that'

exp = explainer.explain_instance(s_json, prediction.predictor, num_features = 10)
words = exp.as_list(label=1)
print(words)

exp = explainer.explain_instance(p_json, prediction.predictor, num_features = 10)
words = exp.as_list(label=1)
print(words)


#
# [('If', 0.16390386602375082), ('word', -0.08405546625806146), ('constraint', -0.07732406414135602), ('activation', -0.07590844811444764), ('repair', -0.0540038291604182), ('and', -0.0525229238487144), ('morphemes', 0.050981928620053234), ('of', 0.045166139453384234), ('syntactic', -0.0450861154417504), ('during', -0.042252321456709103), ('synonyms', -0.03658067636014455), ('accessing', -0.03294164350605486), ('near', 0.03186753985962745), ('Griffin', -0.03016452330556276), ('forms', -0.030002717054465593), ('a', -0.029622284733672847), ('then', 0.02948310885156586), ('in', -0.02681886907882046), ('covert', -0.02444328088328717), ('frequency', 0.02368940517575886), ('Roelofs', 0.023374138235904847), ('1998', -0.02276833539880144), ('transition', -0.021974389029981767), ('from', 0.0209871270
# 5979628), ('rather', -0.020082952684763167), ('effect', -0.018060188767833215),('sentential', 0.01702926331951344), ('i', 0.016059269293892772), ('that', 0.015131875144616997), ('continuous', 0.01389695283408403)]
#
# [('to', 0.07538259335326393), ('C2', -0.023391925765832756), ('is', -0.021813694
# 468966565), ('free', -0.02103700907586576), ('RB', -0.01770718909563708), ('the'
# , 0.014184825115949838), ('participants', -0.013491571069007745), ('item', -0.01
# 2899424291928932), ('bound', -0.0128458589133508), ('identified', 0.012815137489
# 825757), ('second', 0.011928969797925704), ('individuation', -0.0109218048216263
# 28), ('not', 0.010577459179991098), ('However', -0.01011674024419363), ('repeate
# d', -0.009870954758683393), ('unavailable', 0.009800279169372612), ('consciously
# ', -0.009387722674882013), ('C1', -0.008482532996873411), ('1987', -0.0080643060
# 56699548), ('type', 0.007850617420246142), ('but', 0.0077855505249589336), ('tem
# porarily', -0.007565132183168459), ('reported', -0.00747055404303765), ('occurre
# nce', 0.007040701509999923), ('for', -0.007022110891488438), ('abstract', 0.0065
# 4893193054806), ('its', 0.0063669399754552655), ('report', -0.006227851697046469
# 5), ('activates', 0.005720683578042897), ('each', 0.005523718603631709)]
#
#
