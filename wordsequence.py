"""
文本序列化
"""

class WordSequence:
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict={
            self.UNK_TAG:self.UNK,  #表示未知字符
            self.PAD_TAG:self.PAD   #表示填充字符
        }                           #保存词语和对应的数字
        self.count = {}             #统计词频的
    def fit(self,tokens):
        """
        接受句子，消除符号，统计词频(这里的tokens是用来组成字典的）
        :param tokens:[str,str,str]
        :return:None
        """
        prelist = ['\u3000','?','-','T','～','】','>','▽','<', '*','【', '“', '”',  '！', '、','〔', '〕', ')', '：', '—','。','，','？' ,':','《', '》', '·', ' ', '\n', '(', '）','（', '？' '.', '/', '!', '，', '。','★','」','_','「']
        for token in tokens:
            if token not in prelist:
                self.count[token] = self.count.get(token,0)+1
        return self.count

    def build_vocab(self,min=None,max=None,max_feature=None):
        """
        根据条件构造词典
        :param min:最小词频
        :param max: 最大词频
        :param max_features: 最大词语个数
        :return:
        """
        if min is not None:
            self.count = {word:value for word,value in self.count.items() if value>=min}
        if max is not None:
            self.count = {word:value for word,value in self.count.items() if value<=max}
        if max_feature is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_feature]
            self.count = dict(temp)
        for word in self.count:
            self.dict[word] = len(self.dict)                                   # 每个word对应一个数字
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))    # 用zip把dict进行翻转
        return self.inverse_dict

    def transform(self, tokens, max_len=None):
        """
        把句子转化为数字序列
        :param tokens:[str,str,str]
        :return: [int,int,int]
        """
        if max_len is not None:
            if max_len > len(tokens):
                tokens = tokens + [self.PAD_TAG] * (max_len - len(tokens))#填充
            if max_len < len(tokens):
                tokens = tokens[:max_len]                                 #截断
        return [self.dict.get(word,self.UNK) for word in tokens]
    def inverse_transform(self,indices):
        """
        把数字序列转化为字符
        :param indices: [int,int,int]
        :return: [str,str,str]
        """
        return [self.inverse_dict.get(idx) for idx in indices]

    def __len__(self):
        return len(self.dict)












