#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-15 11:32:52
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$
'''
符号说明：
[l]:表示神经网络中的第l层，如a[4]表示第4层的神经网络激活值
x(i):训练集中的第i个训练样本
x(i)<t>:第i个训练样本中序列中第t个元素
a[l]i:第i个训练样本在第l层后的激活值
'''

import numpy as np
from rnn_utils import *

'''实现一个基本的RNN cell计算'''


def rnn_cell_forward(xt, a_prev, parameters):
    """
    参数说明:
	xt shape:(n_x, m) 有m个样本，n_x个特征
	a_prev shape:(n_a, m)  有m个样本，n_a个神经元组成一个RNN cell
	Waa shape: (n_a, n_a)
	Wax shape: (n_a, n_x)
	Wya shape: (n_y, n_a)
	ba shape: (n_a, 1)
	by shape: (n_y, 1)
	a_next shape: (n_a, m)
	yt_pred shape: (n_y, m)
    """
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    temp = np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba
    a_next = np.tanh(temp)
    temp2 = np.dot(Wya, a_next) + by
    yt_pred = softmax(temp2)
    # 使用缓存值，保留a_next, a_prev, xt, parameters
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


np.random.seed(1)  # 设置随机数种子，保证每次产生的随机数都是一样的
xt = np.random.randn(3, 10)
a_prev = np.random.randn(5, 10)
Waa = np.random.randn(5, 5)
Wax = np.random.randn(5, 3)
Wya = np.random.randn(2, 5)
ba = np.random.randn(5, 1)
by = np.random.randn(2, 1)
parameters = {'Wax': Wax, 'Waa': Waa, 'Wya': Wya, 'ba': ba, 'by': by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print('a_next[4] = ', a_next[4])
print('a_next.shape = ', a_next.shape)
print('yt_pred[1] = ', yt_pred[1])
print('yt_pred.shape = ', yt_pred.shape)
print('--------------------------------------')
'''实现RNN 前向传播'''


def rnn_forward(x, a0, parameters):
    '''
    :param x:(n_x, m, T_x)
    :param a0: (n_a, m)
    :param parameters: Waa--(n_a, n_a)    Wax--(n_a, n_x)  Wya--(n_y, n_a)    ba--(n_a, 1) by--(n_y, 1
    :return: a -- (n_a, m, T_x)   y_pred -- (n_y, m, T_x)   caches -- contains (list of caches, x)
    '''
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)
    caches = (caches, x)
    return a, y_pred, caches

np.random.seed(1)
x = np.random.randn(3, 10, 4)
a0 = np.random.randn(5, 10)
Waa = np.random.randn(5, 5)
Wax = np.random.randn(5, 3)
Wya = np.random.randn(2, 5)
ba = np.random.randn(5, 1)
by = np.random.randn(2, 1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))
print('------------------------------------------')

'''LSTM长短时记忆网络中的单个cell'''


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """

    :param xt: xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    :param a_prev: a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    :param c_prev: c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    :param parameters: -- python dictionary containing:
         Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wu -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bu -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :return:
        a_next -- next hidden state, of shape (n_a, m)
        c_next -- next memory state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    """
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    concat = np.zeros((n_a + n_x, m))
    concat[:n_a, :] = a_prev
    concat[n_a:, :] = xt
    ft = sigmoid(np.matmul(Wf, concat) + bf)
    ut = sigmoid(np.matmul(Wu, concat) + bu)
    cct = np.tanh(np.matmul(Wc, concat) + bc)
    c_next = ft*c_prev + ut*cct
    ot = sigmoid(np.matmul(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    yt_pred = softmax(np.matmul(Wy, a_next) + by)
    cache = (a_next, c_next, a_prev, c_prev, ft, ut, ot, cct, xt, parameters)
    return a_next, c_next, yt_pred, cache

np.random.seed(1)
xt = np.random.randn(3, 10)
a_prev = np.random.randn(5, 10)
c_prev = np.random.randn(5, 10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5, 1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5, 1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5, 1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5, 1)
Wy = np.random.randn(2, 5)
by = np.random.randn(2, 1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", c_next.shape)
print("c_next[2] = ", c_next[2])
print("c_next.shape = ", c_next.shape)
print("yt[1] =", yt[1])
print("yt.shape = ", yt.shape)
print("cache[1][3] =", cache[1][3])
print("len(cache) = ", len(cache))
print('-------------------------------------')
'''lstm前向传播'''

def lstm_forward(x, a0, parameters):
    '''
    :param x:Input data for every time-step, of shape (n_x, m, T_x)
    :param a0:Initial hidden state, of shape (n_a, m)
    :param parameters:--------python dictionary containing:
        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wu -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bu -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :return:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    '''

    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    a_next = a0
    c_next = np.zeros(a_next.shape)
    for t in range(T_x):
        a_next, c_next, yt, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        a[:, :, t] = a_next
        y[:, :, t] = yt
        c[:, :, t] = c_next
        caches.append(cache)
    caches = (cache, x)
    return a, y, c, caches


np.random.seed(1)
x = np.random.randn(3, 10, 7)
a0 = np.random.randn(5, 10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5, 1)
Wu = np.random.randn(5, 5+3)
bu = np.random.randn(5, 1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5, 1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5, 1)
Wy = np.random.randn(2, 5)
by = np.random.randn(2, 1)

parameters = {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)
print("a[4][3][6] = ", a[4][3][6])
print("a.shape = ", a.shape)
print("y[1][4][3] =", y[1][4][3])
print("y.shape = ", y.shape)
print("caches[1][1[1]] =", caches[1][1][1])
print("c[1][2][1]", c[1][2][1])
print("len(caches) = ", len(caches))