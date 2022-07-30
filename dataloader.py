"""
完成数据集的加载
"""
from torch.utils.data import DataLoader,Dataset
import os
import config
import torch
from tokenlize import tokenlize
import pickle

class ImdbDataset(Dataset):
    def __init__(self,train,using):
        self.train_data_path = r"C:\Users\Alen\Desktop\gitcode\NLPSA\data\train"
        self.test_data_path = r"C:\Users\Alen\Desktop\gitcode\NLPSA\data\test"
        self.data_path = r"C:\Users\Alen\Desktop\gitcode\NLPSA\data\steam"
        if using:
            self.total_file_path = []                                                   # 所有评论文件的path
            file_name_list = os.listdir(self.data_path)
            file_path_list = [os.path.join(self.data_path, i) for i in file_name_list]
            self.total_file_path.extend(file_path_list)
        else:
            data_path = self.train_data_path if train else self.test_data_path           #把所有文件名放入列表
            temp_data_path = [os.path.join(data_path,"pos"),os.path.join(data_path,"neg")]
            self.total_file_path = []                                                     #所有评论文件的path
            for path in temp_data_path:
                file_name_list = os.listdir(path)
                file_path_list = [os.path.join(path,i)  for i in file_name_list]
                self.total_file_path.extend(file_path_list)
    def __getitem__(self, index):
        file_path=self.total_file_path[index]
        lable_str = file_path.split("\\")[-2]                                              #获取lable
        if lable_str == 'neg':
            lable = 0
        elif lable_str == 'pos':
            lable = 1
        else:
            lable=10000
        tokens = tokenlize(open(file_path,encoding='utf-8').read())                         #获取内容
        return lable,tokens
    def __len__(self):
        return len(self.total_file_path)



def my_collate(batch):
    ws = pickle.load(open('./model/ws.pkl', 'rb'))
    label ,content = list(zip(*batch))
    content = torch.LongTensor([ws.transform(i,max_len=config.max_len) for i in content])
    label = torch.LongTensor(label)
    return label,content


def get_dataloader(train,using,batch_size):
    imdb_dataset = ImdbDataset(train,using)
    data_loader = DataLoader(dataset=imdb_dataset,batch_size=batch_size,shuffle=True,collate_fn=my_collate)
    return data_loader

