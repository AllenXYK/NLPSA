"""
实现分词的方法
"""

import jieba
def tokenlize(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """
    jieba.setLogLevel(jieba.logging.INFO)
    tokens = jieba.lcut(sentence)
    return tokens