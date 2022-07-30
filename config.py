#coding:utf-8
import pickle
import torch

"""
dataset
"""
min_count = 10
max_count = 500
max_feature =15000


"""
dataloader
"""

max_len = 100


"""
model
"""
hidden_size = 128
num_layers = 4
embedding_dim = 100
bidriectional = True
dropout = 0.6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_batch_size = 128
test_batch_size = 128


