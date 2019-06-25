#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-25 17:05:42
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231
# @Version : $Id$

import os
import tensorflow as tf
import numpy as np

"""
	charRNN 是N vs N的循环神经网络，要求输入序列长度等于输出序列长度。
	原理：用已经输入的字母去预测下一个字母的概率。一个句子是hello!,例如输入序列是hello,则输出序列是ello!
	预测时：首先选择一个x1当作起始的字符，然后用训练好的模型得到下一个字符出现的概率。
	根据这个概率选择一个字符输出，然后将此字符当作下一步的x2输入到模型中。依次递推，得到任意长度的文字。
"""


# 注意：输入的单个字母是以one-hot形式进行编码的！
"""
	对中文进行建模时，每一步输入模型的是一个汉字，汉字的种类太多，导致模型太大，一般采用下面的方法进行优化：
		1.取最常用的N个汉字，将剩下的汉字变成单独的一类，用一个<unk>字符来进行标注
		2.在输入时，可以加入一个embedding层，将汉字的one-hot编码转为稠密的词嵌入表示。
		对单个字母不使用embedding是由于单个字母不具备任何的含义，只需要使用one-hot编码即可。单个汉字是具有一定的实际意义的，所以使用embedding层
"""

# 实现RNN的基本单元RNNCell抽象类--------有两种直接使用的子类:BasicRNNCell(基本的RNN)和LSTMCell(基本的LSTM)

"""

RNNCell有三个属性:
	1.类方法call:所有的子类都会实现一个__call__函数，可以实现RNN的单步计算，调用形式为(output,next_state) = __call__(input, state)
	2.类属性state_size:隐藏层的大小，输入数据是以batch_size的形式进行输入的即input=(batch_size, input_size),
	调用__call__函数时隐藏层的形状是(batch_size, state_size),输出层的形状是(batch_size, output_size)
	3.类属性output_size:输出向量的大小
"""


# 定义一个基本的RNN单元
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
print("rnn_cell.state_size:", rnn_cell.state_size)

# 定义一个基本的LSTM的基本单元
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128)
print("lstm_cell.state_size:", lstm_cell.state_size)


lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128)
# batch_size=32, input_size=100
inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = lstm_cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell.__call__(inputs, h0)
print(h1.c)
print(h1.h)

# 对RNN进行堆叠：MultiRNNCell

# 每次调用这个函数返回一个BasicRNNCell


def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)


# 使用MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])
# 得到的RNN也是RNNCell的子类,state_size=(128, 128, 128):三个隐层状态，每个隐层状态的大小是128
print(cell.state_size)

# 32是batch_size, 100是input_size
inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = cell.zero_state(32, np.float32)
output, h1 = cell.__call__(inputs, h0)
print(h1)


# 使用tf.nn.dunamic_rnn按时间展开：相当于增加了一个时间维度time_steps,通过{h0,x1,x2...,xn}得到{h1,h2,h3,...hn}
# 输入数据的格式是(batch_size, time_steps, input_size)
"""
inputs: shape=(batch_size, time_steps, input_size)
initial_state:  shape(batch_size,cell.state_size)  初始状态,一般可以取零矩阵
outputs, state = tf.nn.dynamic_rnn(cell,inputs,initial_state)  
# outputs是time_steps中所有的输出，形状是(batch_size, time_steps, cell.output_size)
# state是最后一步的隐状态，形状是(batch_size,cell.state_size)

注意：输入数据的形状是(time_steps,batch_size, input_size),可以调用tf.nn.dynamic_rnn()函数中设定参数time_major=True
此时，得到的outputs的形状是(time_steps, batch_size, cell.output_size);state的形状不变化
"""