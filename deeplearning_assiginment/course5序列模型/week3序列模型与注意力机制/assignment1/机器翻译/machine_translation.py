#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2
import numpy as np
from nmt_utils import *
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
import matplotlib.pyplot as plt
"""
神经网络机器翻译NMT，将"25th of June,2009"翻译成"2009-06-25",使用的是注意力机制
"""
# 数据集
m = 10000  # 10000个人类可阅读的日期数据和机器可以阅读的日期数据
"""
dataset是由元组组成，每个元组是人类阅读的日期数据与机器可以阅读的日期数据
human_vocab:python的一个字典，将人类可以阅读的日期数据映射成一个整数索引
machine_vocab: python的一个字典，将机器可以阅读的日期数据映射成一个整数索引
inv_machine_vocab:machine_vocab相反的结果，将整数索引映射成一个机器可以阅读的日期数据
"""
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
print(dataset[:10])
print("human_vocab:\n", human_vocab)
print("len(human_vocab)=", len(human_vocab))
print("machine_vocab:\n", machine_vocab)
print("len(machine_vocab)=", len(machine_vocab))  # 0-9再加上"-"字符，所以是11
print("inv_machine_vocab:\n", inv_machine_vocab)
# 数据的预处理，将原始txt文件中的数据映射为整数索引，Tx=30:假设人类可以阅读的日期数据的最大长度
# 如果输入的日期数据大于30，将会对其剪切，机器翻译的日期数据最大长度10
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
print("X:\n", X)
print("Xoh:\n", Xoh)
print("X的形状:", X.shape)  # (m, Tx)
print("Y的形状:", Y.shape)  # (m, Ty)
print("Xoh的形状:", Xoh.shape)  # (m, Tx, len(human_vocab))
print("Yoh的形状:", Yoh.shape)  # (m, Ty, len(machine_vocab))

"""
参数说明:
    X:训练集中已经预处理后的人类可以阅读的日期数据，每个字符都被索引所取代,索引从human_vocab中获取
    Y:训练集中已经预处理后的机器可以阅读的日期数据，每个字符都被索引所取代，索引从machine_vocab中获取
    Xoh:X的one-hot编码形式
    Yoh:Y的one-hot编码形式
    len(human_vocab): 输入词典的大小37 = 26 + 11
    len(machine_vocab): 输出词典的大小11
"""
index = 0
print("原始数据:", dataset[index][0])
print("目标数据:", dataset[index][1])
print("经过预处理之后的结果:")
print(X[index])
print(Y[index])
print(Xoh[index])
print(Yoh[index])
"""
使用attention机制的机器翻译
模型介绍：
1.模型利用两个分离的LSTM，底部的LSTM使用的是Bidirectional LSTM,出现在attention之前Bi-LSTM，经过Tx时间序列；顶部的LSTM是出现在attention之后，经过Ty时间序列
2.顶部的LSTM输入s_t, c_t随时间序列Ty传递；将s_t输入到下一个序列，不使用y_t-1输入到下一个序列；因为不像文本生成，相邻的词汇之间有很强的相关性
3.使用a_t = (a_>t, a_>t)表示Bidirectional LSTM的激活函数值
4.使用RepeatVector去copy s_<t-1>的值，然后使用Concatenation合并s_<t-1>与a_<t>来计算e_<t,t_up>,经过一个softmax来计算注意力权重值
"""

"""
one_step_attention()函数：
根据双向Bi-LSTM去计算a_<t>的值和顶部LSTM的隐藏状态得到的s_<t-1>，两者结合，计算注意力权重，输出是context_<t>向量
注意：这里使用context_<t>，而不使用c_<t>,是为了避免与顶层的LSTM内部记忆变量混淆
model()函数：
实现整个模型，首先给Bi-LSTM输入，然后返回a_<t>的值；之后调用one_step_attention()函数计算出context_<t>，输出给顶层的LSTM，
顶层LSTM的输出经过一个softmax，得到最终的预测值y_hat_<t>
"""
repeator = RepeatVector(Tx)  # 将输入重复Tx次
concatenator = Concatenate(axis=-1)  # 该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量
densor1 = Dense(10, activation="tanh")
densor2 = Dense(1, activation="relu")
activator = Activation(softmax, name="attention_weights")  # 使用自定义的激活函数
dotor = Dot(axes=1)


def one_step_attention(a, s_prev):
    """

    :param a: Bi-LSTM中隐藏状态的输出(m, Tx, 2*n_a)
    :param s_prev: 顶部LSTM的上一时刻的隐藏状态s_<t-1>,形状是(m, n_s)
    :return: context向量，输入给顶层的LSTM
    """
    s_prev = repeator(s_prev)   # 将s_prev进行重复输入，转化为(m, Tx, n_s)的形状，为了下一步与a_<t>合并做准备
    concat = concatenator([a, s_prev])  # 将s_<t-1>与a<t>合并成一个向量，准备计算注意力权重alpha_<t>
    e = densor1(concat)  # 使用densor1去传播concat，使其经过一个小的全连接层，计算中间的能量
    energies = densor2(e)  # 使用densor2去传播e，使其经过一个小的全连接层，计算能量
    alphas = activator(energies)  # 使用activator去计算注意力权重
    context = dotor([alphas, a])  # 使用矩阵乘法，计算context向量
    return context

n_a = 32
n_s = 64  # 顶层LSTM输出的维度
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)   # 最终的预测结果


# 建立整个模型
def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """

    :param Tx: 输入时间序列的长度
    :param Ty: 输出时间序列的长度
    :param n_a: Bi-LSTM的隐藏状态大小
    :param n_s: 顶部LSTM的隐藏状态大小
    :param human_vocab_size: 输入human_vocab的大小
    :param machine_vocab_size: 输出machine_vocab的大小
    :return: model
    """
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name="s0")
    c0 = Input(shape=(n_s,), name="c0")
    s = s0
    c = c0
    outputs = []
    a = Bidirectional(LSTM(n_a, return_sequences=True), name="bidirectional_1")(X)
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model


# 测试上述函数的运行结果
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()
# 编译上述模型
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# 初始化s0 与 c0
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))
# 开始训练
model.fit([Xoh, s0, c0], outputs, epochs=10, batch_size=100)
model.save("models/model.h5")  # 保存模型
model.save_weights("models/weights.h5")  # 保存权重信息
model.load_weights("models/weights.h5")  # 加载权重信息
# 使用实际案例进行测试
EXAMPLES = ['3 May 1979', '5 April 2099', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
            'March 3rd 2001', '1 March 2018']
for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
    source = source.transpose()  # 交互两轴
    source = np.expand_dims(source, axis=0)  # 增加一个轴
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("输入:", example)
    print("预测值:", ''.join(output))

# 可视化注意力权重值
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num=7, n_s=64)
print(attention_map)

