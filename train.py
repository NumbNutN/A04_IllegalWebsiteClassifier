# coding: UTF-8
import os
import time
import torch
import numpy as np
from train_eval_utils import train, init_network, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=False, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 分别用于对标题、关键词、特征文本进行训练
if __name__ == '__main__':
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    dataset = 'IllegalWebsite/title'                  # 数据集

    model_name = 'bert'                         #args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)                        #为CPU设置固定种子
    torch.cuda.manual_seed_all(1)               #为所有GPU设置固定种子
    torch.backends.cudnn.deterministic = True   #保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

    #test
    #model = x.Model(config).to(config.device)
    #test(config, model,
