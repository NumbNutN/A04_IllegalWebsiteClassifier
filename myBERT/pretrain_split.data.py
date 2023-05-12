from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
import sys
sys.path.append("../../")
from tool import feature_extraction_tool as fet

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


#读取文档
ori_test_texts = fet.read_csv_context(
                                filename="/A04/bert_data/all_content_split_test.csv",
                                row_range = range(3000000),
                                col = 0)

ori_test_labels = fet.read_csv_context(
                                filename="/A04/bert_data/all_content_split_test.csv",
                                row_range = range(3000000),
                                col = 1)

test_texts = []


test_labels = []
for idx in range(len(ori_test_labels)):
    if(ori_test_labels[idx] != '1'and ori_test_labels[idx] != '12'):
        test_texts.append(ori_test_texts[idx])
        test_labels.append(ori_test_labels[idx])

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

test_labels = [int(ids)-2 for ids in test_labels]
test_labels = torch.tensor(test_labels)



# 定义测试数据集和数据加载器
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_sampler = RandomSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

model = torch.load("/A04/IllegalWebsiteClassifier/myModel/bilstm_bert_split.pt")
evaluate(model,test_dataloader)