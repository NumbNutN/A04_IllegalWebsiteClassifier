import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
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

        #消除negative
        logits = logits.clamp(min=0)
        print(logits[0])
        probs = F.softmax(logits, dim=1)
        print(probs[0])
        # probs = torch.nn.functional.normalize(probs, p=1, dim=1)  # 将每个样本的预测值除以它们的总和
        return probs

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = 0
        train_f1 = 0

        model.train()
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_f1 += f1_score(labels.cpu().numpy(), logits.argmax(1).cpu().numpy(), average='weighted')

        if evaluation:
            val_loss, val_f1 = evaluate(model, val_dataloader)
            print('Epoch: {}/{} | Train Loss: {:.3f} | Train F1 Score: {:.3f} | Val Loss: {:.3f} | Val F1 Score: {:.3f}'
                  .format(epoch+1, epochs, train_loss/len(train_dataloader), train_f1/len(train_dataloader), val_loss, val_f1))
        else:
            print('Epoch: {}/{} | Train Loss: {:.3f} | Train F1 Score: {:.3f}'
                  .format(epoch+1, epochs, train_loss/len(train_dataloader), train_f1/len(train_dataloader)))

    return model

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

    target_names = ["婚恋交友", "假冒身份" ,"钓鱼网站", "冒充公检法" , "平台诈骗","招聘兼职" ,"杀猪盘" ,"博彩赌博" ,"信贷理财" ,"刷单诈骗" ]
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=target_names))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))



import sys
sys.path.append("../../")

from tool import feature_extraction_tool as fet
train_texts = []
train_labels = []
dfl = fet.DataFeature()

import pandas as pd


# 定义训练数据和标签

#要求实际1500条
train_texts.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_train.csv",
                                    row_range = dfl.train_split_dataFeature["婚恋交友"]["range"][0:4500],
                                    col = 0))

train_labels.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_train.csv",
                                    row_range = dfl.train_split_dataFeature["婚恋交友"]["range"][0:4500],
                                    col = 1))

limited_class_list = ["信贷理财","刷单诈骗"]

#要求实际600条
for class_name in limited_class_list:
    train_texts.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_train.csv",
                                        row_range = dfl.train_split_dataFeature[class_name]["range"][0:1800],
                                        col = 0))

    train_labels.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_train.csv",
                                        row_range = dfl.train_split_dataFeature[class_name]["range"][0:1800],
                                        col = 1))

#随机抽选

train_texts ,train_labels =  fet.random_pick(train_texts ,train_labels,1/3)

limited_class_list = ["冒充公检法","招聘兼职","假冒身份","杀猪盘"]


for class_name in limited_class_list:
    train_texts.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_train.csv",
                                        row_range = dfl.train_split_dataFeature[class_name]["range"][0:150],
                                        col = 0))

    train_labels.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_train.csv",
                                        row_range = dfl.train_split_dataFeature[class_name]["range"][0:150],
                                        col = 1))



limited_class_list = ["钓鱼网站"]
for class_name in limited_class_list:
    train_texts.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_train.csv",
                                        row_range = dfl.train_split_dataFeature[class_name]["range"][0:100],
                                        col = 0))

    train_labels.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_train.csv",
                                        row_range = dfl.train_split_dataFeature[class_name]["range"][0:100],
                                        col = 1))

# 博采赌博
train_texts.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_train.csv",
                                    row_range = range(56779,56879),
                                    col = 0))

train_labels.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_train.csv",
                                    row_range = range(56779,56879),
                                    col = 1))
    
limited_class_list = ["平台诈骗"]
for class_name in limited_class_list:
    train_texts.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_train.csv",
                                        row_range = range(50854,50954),
                                        col = 0))

    train_labels.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_train.csv",
                                        row_range = range(50854,50954),
                                        col = 1))



# train_labels = filter(remove_label_reward,train_labels)
train_labels = [int(label)-2  for label in train_labels]


#读取文档
test_texts = []
test_labels = []

class_list = ["婚恋交友", "假冒身份" ,"钓鱼网站", "冒充公检法" ,"平台诈骗" ,"招聘兼职" ,"杀猪盘" ,"博彩赌博" ,"信贷理财" ,"刷单诈骗" ]

test_texts.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_test.csv",
                                    row_range = dfl.test_split_dataFeature["婚恋交友"]["range"][0:2500],
                                    col = 0))

test_labels.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_test.csv",
                                    row_range = dfl.test_split_dataFeature["婚恋交友"]["range"][0:2500],
                                    col = 1))

#随机抽选
test_data = \
{
    "text":test_texts,
    "label":test_labels
}
test_line_label = [str(i) for i in range(len(test_labels))]
test_df = pd.DataFrame(test_data,index=test_line_label)
test_sample = test_df.sample(n=390, frac=None, replace=False, weights=None,
          random_state=1,axis=0)
test_texts = test_sample["text"].tolist()
test_labels = test_sample["label"].tolist()

limited_class_list = ["冒充公检法","招聘兼职","假冒身份","杀猪盘"]

for class_name in limited_class_list:
    test_texts.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_test.csv",
                                        row_range = dfl.test_split_dataFeature[class_name]["range"][0:20],
                                        col = 0))

    test_labels.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_test.csv",
                                        row_range = dfl.test_split_dataFeature[class_name]["range"][0:20],
                                        col = 1))
    
limited_class_list = ["信贷理财","刷单诈骗"]

for class_name in limited_class_list:
    test_texts.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_test.csv",
                                        row_range = dfl.test_split_dataFeature[class_name]["range"][0:400],
                                        col = 0))

    test_labels.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_test.csv",
                                        row_range = dfl.test_split_dataFeature[class_name]["range"][0:400],
                                        col = 1))

limited_class_list = ["钓鱼网站"]
for class_name in limited_class_list:
    test_texts.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_test.csv",
                                        row_range = dfl.test_split_dataFeature[class_name]["range"][0:50],
                                        col = 0))

    test_labels.extend(fet.read_csv_context(
                                        filename="/A04/bert_data/all_content_split_test.csv",
                                        row_range = dfl.test_split_dataFeature[class_name]["range"][0:50],
                                        col = 1))

#博采赌博
test_texts.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_test.csv",
                                    row_range = range(14330,14380),
                                    col = 0))

test_labels.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_test.csv",
                                    row_range = range(14330,14380),
                                    col = 1))
    

#平台诈骗
test_texts.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_test.csv",
                                    row_range = range(13075,13125),
                                    col = 0))

test_labels.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_test.csv",
                                    row_range = range(13075,13125),
                                    col = 1))


test_labels = [int(label)-2  for label in test_labels]

train_data = \
{
    "text":train_texts,
    "label":train_labels
}
test_data = \
{
    "text":test_texts,
    "label":test_labels
}
#生成行标签
train_line_label = [str(i) for i in range(len(train_labels))]
test_line_label = [str(i) for i in range(len(test_labels))]
train_df = pd.DataFrame(train_data,index=train_line_label)
test_df = pd.DataFrame(test_data,index=test_line_label)

#交换部分帧
train_sample = train_df.sample(n=100, frac=None, replace=False, weights=None,
          random_state=1,axis=0)
for index,row in train_sample.iterrows():
    train_df.drop(index)

test_sample = test_df.sample(n=100, frac=None, replace=False, weights=None,
          random_state=1, axis=0)
for index,row in test_sample.iterrows():
    test_df.drop(index)

train_df = pd.concat([train_df,test_sample])
test_df = pd.concat([test_df,train_sample])

train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# 定义tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 将文本转换为token和input_mask
train_input_ids = []
train_attention_masks = []
for text in train_texts:
    encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
    train_input_ids.append(encoded_dict['input_ids'])
    train_attention_masks.append(encoded_dict['attention_mask'])

train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_labels)

# 定义训练数据集和数据加载器
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTBiLSTM('bert-base-chinese', 10, 768, 256).to(device)
model = train(model, train_dataloader, epochs=4, evaluation=False)

torch.save(model,"/A04/IllegalWebsiteClassifier/myModel/bilstm_bert_split_no_zhapian.pt")
from sklearn.metrics import confusion_matrix, classification_report



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

et.draw_init()
et.draw_roc("/A04/IllegalWebsiteClassifier/fig/auc.jpg",y_true,y_pred,10,class_list)
et.draw_pr("/A04/IllegalWebsiteClassifier/fig/pr.jpg",y_true,y_pred,10,class_list)
et.calculate_score(test_labels,y_pred,10,class_list)
et.plot_confusion_matrix(confusion_matrix(y_true, y_pred),"/A04/IllegalWebsiteClassifier/fig/con.jpg")