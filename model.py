"""
定义模型
"""

from dataloader import get_dataloader
import config
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import os
import numpy as np
import pickle


class MyModel(nn.Module):
    def __init__(self):
        ws = pickle.load(open('./model/ws.pkl', 'rb'))
        super(MyModel,self).__init__()
        self.embedding = nn.Embedding(len(ws),config.embedding_dim)
        self.lstm = nn.LSTM(input_size= config.embedding_dim,hidden_size=config.hidden_size,num_layers=config.num_layers,
                    batch_first=True, bidirectional=config.bidriectional,dropout=config.dropout)
        self.fc1 = nn.Linear(config.hidden_size*2,32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self,input):
        #input：[batch_size,max_len]
        #return:
        x = self.embedding(input)#进行embedding操作，形状：[batch_size,max_len,embedding_dim]
        x,(h_n,c_n) = self.lstm(x)#隐藏状态和细胞状态h_n,c_n形状[num_layer*2,batch_size,hidden_size]
        #x形状：[batch_size,max_len,hidden_size*num_layer]
        #获取两个方向最后一次的Output进行concat
        output_f = h_n[-2,:,:]#正向最后一次输出
        output_b = h_n[-1,:,:]#反向最后依次输出
        output = torch.cat([output_f,output_b],dim=-1)#[batch_size,hidden_size*2]
        output = self.fc1(output)
        output = F.relu(output)
        out = self.fc2(output)
        return F.log_softmax(out,dim=-1)

model = MyModel().to(config.device)
optimizer = Adam(model.parameters(),0.0001)
if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))


def train(Epoch,SumEpoch):
    for idx,(target,input) in enumerate(get_dataloader(train=True,using=False, batch_size=config.train_batch_size)):
        target=target.to(config.device)
        input=input.to(config.device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if ( idx -1 ) % 30 == 0:
            print('Epoch[{}/{}],loading:{}%,loss:{:8f}'.format(Epoch, SumEpoch,'99', loss.data))

        if ( idx -1 ) % 240 == 0:
            torch.save(model.state_dict(),'./model/model.pkl')
            torch.save(optimizer.state_dict(),'./model/optimizer.pkl')


def eval():
    loss_list=[]
    acc_list=[]
    for idx,(target,input) in enumerate(get_dataloader(train=False,using=False,batch_size=config.test_batch_size)):
        target = target.to(config.device)
        input = input.to(config.device)
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss.cpu().item())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())
    print("total loss,acc:",np.mean(loss_list),np.mean(acc_list))

def use():

    for idx, (target, input) in enumerate(get_dataloader(train=False, using=True, batch_size=1)):
        input = input.to(config.device)
        with torch.no_grad():
            print(input)
            output = model(input)
            print(output)
            pred = output.max(dim=-1)[1]
            print(pred.item())



