"""
准备数据
"""

from wordsequence import WordSequence
from tokenlize import tokenlize
import os
import pickle
import config


ws = WordSequence()
data_path = r"C:\Users\Alen\Desktop\gitcode\NLPSA\data\train"
temp_data_path = [os.path.join(data_path, "pos"), os.path.join(data_path, "neg")]
for path in temp_data_path:
    file_name_list = os.listdir(path)
    for i in file_name_list:
        if i.endswith('.txt'):
            file_path_list = os.path.join(path,i)
            sentence = tokenlize(open(file_path_list,encoding='utf-8').read())
            ws.fit(sentence)
ws.build_vocab(min=config.min_count,max = config.max_count,max_feature=config.max_feature)
# print(ws.inverse_dict)
pickle.dump(ws, open('./model/ws.pkl', 'wb'))
print(len(ws))
# ret = ws.transform(["男主角", "智商", "失望", "背景", "配音", "失望", "主演", "失望"], max_len=10)
# print(ret)
ret = [10181,   120,     0,     0,     0,     0,     0,  6848,     0,     0,
             0,  1553,  3115,  2439,     0,     0,   107,   141,  1464,     0,
          1133,     0,     0,     0,     0,  3986,     0,  1257,     0,  6866,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,    19,   145,     0,     0,     0,     0,   224,     0,     0,
         12319,     0,     0,     0,     0,     0,     0,     0,     0,  2360,
             0,     0,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1]
ret = ws.inverse_transform(ret)
print(ret)

