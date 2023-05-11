conda create -n BERTGPU python=3.8
conda activate BERTGPU

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -U scikit-learn
pip install tqdm
pip install pytorch-pretrained-bert

self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
config.tokenizer.convert_tokens_to_ids(token)



https://blog.csdn.net/Defiler_Lee/article/details/126490287
bert-base-chinese


https://zhuanlan.zhihu.com/p/448148614
bert-base-cased