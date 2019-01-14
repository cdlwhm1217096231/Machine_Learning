#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2
from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

X, Y, n_values, indices_values = load_music_utils()
print('输入X的维度信息:', X.shape)      # （m, T_x, 78）
print('训练样本数量:', X.shape[0])  # 60个样本
print('序列长度T_x:', X.shape[1])   # 输入时间序列长度T_x=30
print('总共有多少唯一值:', n_values)  # indices_values:即n_values=78的字典索引值，范围是0-77
print('输出Y的维度信息:', Y.shape)   # (T_y, m, 78).输出时间序列长度T_y=30,则T_x=T_y
print('*'*40)
# 构建模型
n_a = 64  # LSTM cell中的隐藏单元个数
reshapor = Reshape((1, 78))
LSTM_cell = LSTM(n_a, return_state=True)
densor = Dense(n_values, activation='softmax')


# 实现djmodel()模型
def djmodel(T_x, n_a, n_values):
    X = Input(shape=(T_x, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    outputs = []  # 1.创建一个空的list，保存每次迭代输出的结果
    for t in range(T_x):   # 2.1 开始循环
        x = Lambda(lambda x: X[:, t, :])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model

# 定义模型
model = djmodel(T_x=30, n_a=64, n_values=78)
# 编译与训练模型
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 初始化a0、c0
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
model.fit([X, a0, c0], list(Y), epochs=1000)
print('*'*40)


# 使用已经训练好的模型，生成序列的值
def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, T_y=100):
    x0 = Input(shape=(1, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    outputs = []
    for t in range(T_y):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        x = Lambda(one_hot)(out)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    return inference_model


inference_model = music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, T_y=100)
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


# 使用上面的inference_model模型,预测下一个值
def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, c_initializer = c_initializer):
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(np.array(pred), axis=-1)
    results = to_categorical(indices, num_classes=x_initializer.shape[-1])
    return indices, results


results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))
# 生成音乐
out_stream = generate_music(inference_model)
