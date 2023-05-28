import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F

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
    
def BERT_content_predict(pre_trained_model_path:str,content:str,class_list:'list[str]'):
    model = torch.load(pre_trained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    encoded_dict = tokenizer.encode_plus(content, add_special_tokens=True, max_length=128, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids'].to(device)
    attention_masks = encoded_dict['attention_mask'].to(device)

    with torch.no_grad():

        out = model(input_ids,attention_masks)
        _, pre = torch.max(out.data, 1)
        return class_list[pre.item()]
    
BERT_content_predict("/A04/IllegalWebsiteClassifier/myModel/bilstm_bert_split_no_zhapian.pt",
                     "诈骗诈骗诈骗",
                     ["婚恋交友", "假冒身份" ,"钓鱼网站", "冒充公检法" , "平台诈骗","招聘兼职" ,"杀猪盘" ,"博彩赌博" ,"信贷理财" ,"刷单诈骗" ])

