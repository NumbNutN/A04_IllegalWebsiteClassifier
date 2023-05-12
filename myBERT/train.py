import torch

from transformers import BertModel, BertTokenizer, BertForSequenceClassification


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from keras.metrics import accuracy

model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=10)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')



def convert_to_text_label(filename:str):
    with open(filename) as fs:
        text = []
        label = []
        lines = fs.readlines()
        for line in lines:
            split_cnt = line.split('\t')
            text.append(split_cnt[0])
            label.append(int(split_cnt[1]))
    return text,label

def train(model, train_dataloader, validation_dataloader, epochs=4, evaluation=False):
    
    # 定义优化器和学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # 定义学习率衰减器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        running_loss = 0.0
        model.train()

        for batch in tqdm(train_dataloader, desc='Training'):
            # 将输入数据放到GPU上
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}

            optimizer.zero_grad()

            outputs = model(**inputs)

            loss = criterion(outputs.logits, inputs['labels'])
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        # 计算验证集上的损失和准确率
        if evaluation:
            model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in tqdm(validation_dataloader, desc='Evaluating'):
                batch = tuple(t.to(device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}

                    outputs = model(**inputs)
                    tmp_eval_loss = criterion(outputs.logits, inputs['labels'])
                    eval_loss += tmp_eval_loss.mean().item()

                    preds = outputs.logits.detach().cpu().numpy()
                    label_ids = inputs['labels'].cpu().numpy()
                    tmp_eval_accuracy = accuracy(preds, label_ids)

                    eval_accuracy += tmp_eval_accuracy
                    nb_eval_steps += 1

            print("Validation loss: {}".format(eval_loss / nb_eval_steps))
            print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_examples))

        scheduler.step()

    return model

# 定义训练数据和标签
train_texts = ['第一篇文本', '第二篇文本', ...]
train_labels = [0, 1, ...] # 标签应该是整数形式，从0开始

train_texts = []
train_labels = []

new_text,new_label = convert_to_text_label("./myBERT/train.txt")
train_texts.extend(new_text)
train_labels.extend(new_label)
new_text,new_label = convert_to_text_label("./myBERT/test.txt")
train_texts.extend(new_text)
train_labels.extend(new_label)
new_text,new_label = convert_to_text_label("./myBERT/dev.txt")
train_texts.extend(new_text)
train_labels.extend(new_label)

train_labels = [int(ids)-2 for ids in train_labels]

# 将文本转换为模型所需的格式
train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
train_labels = torch.tensor(train_labels)

# 将数据和标签打包成dataset
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 定义训练数据的dataloader
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16)

# 定义验证数据的dataloader
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=16)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型和数据移动到设备上
model.to(device)
train_dataloader.to(device)
val_dataloader.to(device)




# 开始训练
model = train(model, train_dataloader, val_dataloader, epochs=4, evaluation=True)
