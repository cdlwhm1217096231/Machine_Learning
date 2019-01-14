#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2
from w2v_utils import *
import numpy as np
import os
"""
由于词嵌入是非常消耗计算资源的，大多数情况下都会加载一个已经预训练好的词嵌入层
"""
# 加载词向量，使用的是50维的Glove词向量来表示每个单词
words, word_to_vec_map = read_glove_vecs("data/glove.6B.50d.txt")
# words:词汇表中的单词，word_to_vec_map:将词汇表中的单词映射成的Glove向量形式
"""
由于采用one-hot编码的单词，不能捕捉到单词之间的相似性，而Glove 向量能提供关于单个单词含义更有用的信息
下面让我们看一下，如何使用Glove向量对比两个单词之间的相似度
衡量两个单词之间的相似性，使用的是余弦相似度Cosine similarity(u,v) = u点乘v /(||u||2 ||v||2) = cos(theta)
如果两个单词之间非常相似，则夹角theta将会很小；此时cos(theta)接近等于1；
"""


# 计算余弦相似度的函数
def cosine_similarity(u, v):
    distance = 0
    dot = np.dot(u, v)  # u, v之间的点乘，是向量的内积
    norm_u = np.sqrt(np.sum(u**2))
    norm_v = np.sqrt(np.sum(v**2))
    cosine_similarity = dot / (norm_u * norm_v)
    return cosine_similarity

# 上述函数的测试结果
father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]
print("father与mother之间的余弦相似度:", cosine_similarity(father, mother))
print("ball与crocodile之间的余弦相似度:", cosine_similarity(ball, crocodile))
print("france-paris与rome-italy之间的余弦相似度:", cosine_similarity(france-paris, rome-italy))
"""
单词类比任务，如 a is to b as c is to d.实际中是man is to woman as king is to queue,方法：e_b - e_a = e_d - e_c
"""


# 计算单词类比性的函数
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    # 将单词都转化为小写字母
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    # 将单词转为词嵌入格式glove向量
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    words = word_to_vec_map.keys()
    max_cosine_sim = -100  # 初始化最大相似度
    best_word = None  # 使用None来初始化最相似的单词
    for w in words:
        if w in [word_a, word_b, word_c]:  # 避免最相似的单词出现在输入的单词中
            continue
        cosine_sim = cosine_similarity(e_b-e_a, word_to_vec_map[w]-e_c)
        if cosine_sim > max_cosine_sim:  # 如果计算出来的余弦相似度大于当前初始化的最大余弦相似度，则把当前值赋值给max_cosine_sim
            max_cosine_sim = cosine_sim
            best_word = w  # 将当前单词赋值给best_word,这个单词就是与e_c最相似的单词
    return best_word

# 测试上述的函数
triads_to_try = [("italy", "italian", "spain"), ("india", "delhi", "japan"), ("man", "woman", "boy"), ("small", "smaller", "large")]
for triad in triads_to_try:
    print("{}-->{}::{}-->{}".format(*triad, complete_analogy(*triad, word_to_vec_map)))

"""
消除词向量之间的偏差：如性别上的偏差man--->woman  vs  programmer-->teacher
"""
g = word_to_vec_map["woman"] - word_to_vec_map["man"]
print(g)
print("列表中的姓名与它们构造向量g之间的相似性:")
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
for name in name_list:
    print(name, cosine_similarity(word_to_vec_map[name], g))
print("其他案例：")
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist',
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for word in word_list:
    print(word, cosine_similarity(word_to_vec_map[word], g))
