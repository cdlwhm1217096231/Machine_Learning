#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2


from emo_utils import *
import emoji
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Layer, Dense, Dropout, LSTM, Activation, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.100d.txt')
X_train, Y_train = read_csv('data/train_emoji.csv')  # 训练集数据
X_test, Y_test = read_csv('data/tesss.csv')   # 测试集数据
maxLen = len(max(X_train, key=len).split())
print(maxLen)


# 将每个单词转化为50维的glove向量，之后将每个单词组合成一个句子，返回的是一个矩阵，shape的形状是(m, max_len)
def sentences_to_indices(X, word_to_index, max_len):
    """

    :param X: 输入的一个句子
    :param word_to_index: 将每个句子中的每个单词转为相应的索引，索引值是根据语料库中的单词位置确定的
    :param max_len: 一个句子中含有的单词最大个数
    :return: X_indicies
    """
    m = X.shape[0]  # 样本数量，即句子的数量
    X_indices = np.zeros((m, max_len))  # 将每个单词转为word_to_index后，再转化为glove中的50维向量，详细见图片embedding1.png
    for i in range(m):
        sentence_words = X[i].lower().split()   # 将每个单词转为小写，之后再切割成单个单词
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j += 1
    return X_indices

# 上面函数的测试结果
X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
print("X1 =", X1)
print("X1_indices =", X1_indices)
print("==================================================")


# 定义预训练好的词嵌入层
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1  # 查询Embedding层的API input_dim=vocab_len：输入数据最大下标 + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # word_to_vec_map的值是glove向量返回的，所以每行表示一个单词，单词是由50维向量表示
    emb_matrix = np.zeros((vocab_len, emb_dim))  # 初始化一个词嵌入矩阵
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]  # 将glove向量返回，存放在词嵌入矩阵中
    # 定义词嵌入层
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)  # input_dim=vocab_len,输入数据最大下标+1，设置trainable=False使得这个编码层不可再训练。
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer

# 上述函数测试结果
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
print("=========================================================")


# 利用Keras搭建emojify的LSTM模型
def emojify_v2(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype="int32")  # 输入层
    # 使用预训练模型，构建词嵌入层
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128)(X)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation("softmax")(X)
    model = Model(inputs=sentence_indices, output=X)
    return model

model = emojify_v2((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C=5)
model.fit(X_train_indices, Y_train_oh, epochs=100, batch_size=32, shuffle=True)  # 开始训练
# 测试集上的精度
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C=5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print("在测试集上的精度:", acc)
# 显示错误标签的样本
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('真实的表情:'+ label_to_emoji(Y_test[i]) + '预测的表情: '+ X_test[i] + label_to_emoji(num).strip())
# 运行自己的测试样本
x_test = np.array(['she is so cute'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
