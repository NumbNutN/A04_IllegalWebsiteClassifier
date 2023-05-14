from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score

class BERTBiLSTM(nn.Module):
    def __init__(self, bert_path, num_labels, hidden_size, lstm_hidden_size):
        super(BERTBiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(2*lstm_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        h_n = self.dropout(h_n)
        logits = self.classifier(h_n)
        return logits
    
def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input_ids, attention_masks, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_masks)
            _, predicted = torch.max(outputs, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    target_names = ["婚恋交友", "假冒身份" ,"钓鱼网站", "冒充公检法" ,"平台诈骗" ,"招聘兼职" ,"杀猪盘" ,"博彩赌博" ,"信贷理财" ,"刷单诈骗" ]
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=target_names))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))



# 定义训练数据和标签
from matplotlib.font_manager import FontManager
import subprocess

mpl_fonts = set(f.name for f in FontManager().ttflist)

print('all font list get from matplotlib.font_manager:')
for f in sorted(mpl_fonts):
    print('\t' + f)

import sys
sys.path.append("../../")

from tool import feature_extraction_tool as fet
test_texts = []
test_labels = []
dfl = fet.DataFeature()

class_list = ["婚恋交友", "假冒身份" ,"钓鱼网站", "冒充公检法" ,"平台诈骗" ,"招聘兼职" ,"杀猪盘" ,"博彩赌博" ,"信贷理财" ,"刷单诈骗" ]

for class_name in class_list:
    test_texts.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_train.csv",
                                    row_range = dfl.train_split_dataFeature[class_name]["range"][0:100],
                                    col = 0))

    test_labels.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_train.csv",
                                    row_range = dfl.train_split_dataFeature[class_name]["range"][0:100],
                                    col = 1))
    


# train_labels = filter(remove_label_reward,train_labels)
test_labels = [int(label)-2 for label in test_labels]

# 定义tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 将文本转换为token和input_mask
test_input_ids = []
test_attention_masks = []
for text in test_texts:
    encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
    test_input_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])

test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.tensor(test_labels)


# 定义测试数据集和数据加载器
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_sampler = RandomSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

model = torch.load("/A04/IllegalWebsiteClassifier/myModel/bilstm_bert_split.pt")
evaluate(model,test_dataloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for input_ids, attention_masks, labels in test_dataloader:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_masks)
        #_, predicted = torch.max(outputs, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.tolist())

import numpy as np
test_labels = np.array(test_labels)
y_pred = np.array(y_pred)
from tool import evaluation_tool as et


# for i in range(len(test_labels)):
#     print(np.argmax(y_pred[i]),  y_true[i])
et.draw_roc("/A04/IllegalWebsiteClassifier/fig/no_split_auc.jpg",y_true,y_pred,10,class_list)
et.draw_pr("/A04/IllegalWebsiteClassifier/fig/no_split_pr.jpg",y_true,y_pred,10,class_list)