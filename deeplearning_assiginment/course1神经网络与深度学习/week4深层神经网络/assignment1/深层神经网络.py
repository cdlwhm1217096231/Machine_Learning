#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-17 14:13:43
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import os
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
"""
上标[l]:表示神经网络的第几层
上标(i):表示第i个实例
下标i:表示一个向量中的第i个分量，如a[l]_i:表示第l层中的第i个激活单元
"""

plt.rcParams["figure.figsize"] = (5.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"
np.random.seed(1)


# 1.初始化
def initialize_parameters(n_x, n_h, n_y):
    """
            n_x:输入的特征数量
            n_h：隐藏层神经元的个数
            n_y:输出层神经元的个数
            W1：第一层权重的形状(n_h, n_x)
            b1:第一层偏置的形状(n_h, 1)
            W2: 第二层权重的形状(n_y, n_h)
            b2:第二层偏置的形状(n_y, 1)
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }
    return parameters


# 测试上述函数
parameters = initialize_parameters(2, 2, 1)
print("W1=", parameters["W1"])
print("b1=", parameters["b1"])
print("W2=", parameters["W2"])
print("b2=", parameters["b2"])


# 2.L层神经网络参数初始化
def initialize_parameters_deep(layer_dims):
    """
            layer_dims:每层神经网络的神经元个数
            返回深层神经网络的每个层的参数W, b
            Y = WX + b
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters["W" +
                   str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters["b" +
                   str(i)] = np.zeros((layer_dims[i], 1))

        assert(parameters['W' + str(i)].shape ==
               (layer_dims[i], layer_dims[i - 1]))
        assert(parameters['b' + str(i)].shape == (layer_dims[i], 1))
    return parameters


# 测试上述函数
parameters = initialize_parameters_deep([5, 4, 3])
print("W1=", parameters["W1"])
print("b1=", parameters["b1"])
print("W2=", parameters["W2"])
print("b2=", parameters["b2"])


# 3.线性前向传播
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)
print("Z=", Z)


# 4.线性激活后的前向传播
def linear_activation_forwrd(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


# 测试上述函数
A_prev, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forwrd(
    A_prev, W, b, activation="sigmoid")
print("With sigmoid: A=", str(A))
A, linear_activation_cache = linear_activation_forwrd(
    A_prev, W, b, activation="relu")
print("With ReLU: A=", str(A))


# 5.L_model_forward()函数
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forwrd(
            A_prev, parameters["W" + str(i)], parameters["b" + str(i)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forwrd(
        A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches


X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print("AL=", AL)
print("length of caches：", len(caches))


# 6. 代价函数
def compute_cost(AL, Y):
    """
            AL:预测类别所对应的概率
            Y：真实的类别标签
    """
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * (np.log(1 - AL)))
    cost = np.squeeze(cost)  # 降低维度
    assert (cost.shape == ())
    return cost


# 测试上述函数
Y, AL = compute_cost_test_case()
print("cost=", compute_cost(AL, Y))


# 7.反向传播过程
