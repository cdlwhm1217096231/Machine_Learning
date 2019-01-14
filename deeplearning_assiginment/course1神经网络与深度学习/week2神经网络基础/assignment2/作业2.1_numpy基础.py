#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2
import numpy as np
import math


# sigmoid函数
def basic_sigmoid(x):
    """
    使用math实现sigmoid函数
    :param x: 一个标量
    :return: sigmoid(x)
    """
    s = 1 / (1 + math.exp(-x))
    return s
# 测试上述函数的结果
print(basic_sigmoid(3))
"""深度学习中，主要使用的是向量，所以上面的仅针对单个标量的情况用的很少，下面是使用numpy达到向量化的实现过程"""
x = np.array([1, 2, 3])
print(np.exp(x))
# 向量与标量之间的运算
x = np.arange(1, 4)
print(x + 3)


# 使用numpy实现sigmoid函数
def sigmoid(x):
    """
    使用numpy实现sigmoid函数
    :param x: x可以是标量或者向量
    :return: sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s
# 测试上述函数的结果
x = np.arange(1, 4)
print(sigmoid(x))


# 计算sigmoid函数的梯度(导数)
def sigmoid_derivative(x):
    """
    求sigmoid函数的梯度
    :param x: 输入是一个标量或者向量
    :return: sigmoid函数的梯度
    """
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds
# 测试上述函数的结果
x = np.arange(1, 4)
print(sigmoid_derivative(x))
# 变换numpy中数组的形状
"""
x.shape: 获取向量或者矩阵的维度信息
x.reshape: 将向量或者矩阵的形状转化为另一种形状
"""


def image2vector(image):
    """
    将彩色图片转化为一个向量即:(n_w, n_h, n_c=3)变成(n_c*n_h*n_w,1)
    :param image: 彩色图片
    :return:返回一个向量
    """
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v

# 测试上述函数
image = np.array([
    [
        [0.67826139, 0.29380381],
        [0.90714982, 0.52835647],
        [0.4215251, 0.45017551]
    ],

    [
        [0.92814219, 0.96677647],
        [0.85304703, 0.52351845],
        [0.19981397, 0.27417313]
    ],

    [
        [0.60659855, 0.00533165],
        [0.10820313, 0.49978937],
        [0.34144279, 0.94630077]
    ]
])
print(image.shape)
print("转换后的样子:")
print(image2vector(image))
print(image2vector(image).shape)
print(image2vector(image).ndim)
"""
正则化: x / ||x||
"""


def normalize_rows(x):
    """
    正则化向量或矩阵
    :param x: 一个矩阵
    :return: 正则化后的矩阵
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)  # axis=1,按行方向，求||x||的值
    x = x / x_norm
    return x
# 测试上述函数的结果
x = np.array([[0, 3, 4], [1, 6, 4]])
print("正则化后的结果:\n", normalize_rows(x))
"""
广播与softmax
广播:是numpy中不同维度向量或数组之间运算的操作
"""


def softmax(x):
    """
    计算矩阵X的每行的sotfmax
    :param x: 一个矩阵
    :return: softmax之后的结果
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum   # 广播机制
    return s
# 测试上述函数
x = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])
print("softmax(x):\n", softmax(x))
"""
vectorization向量化操作
"""
import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
# 传统方法
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print("dot=", dot)
print("传统方法计算时间:", str(1000*(toc-tic)) + "ms")
# 向量化的方法
tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()
print("doc=", dot)
print("向量化计算方法:", str(1000*(toc-tic)) + "ms")
"""
实现L1和L2损失函数
L1损失函数：np.sum(|y - y_pred|)
L2损失函数: np.dot((y - y_pred), (y - y_pred).T)  # 矩阵相乘
"""


# L1损失函数
def L1(y_pred, y):
    loss = np.sum(np.abs(y_pred - y))
    return loss

y_pred = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(y_pred, y)))


# L2损失函数
def L2(y_pred, y):
    loss = np.dot((y - y_pred), (y - y_pred).T)
    return loss
y_pred = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(y_pred, y)))