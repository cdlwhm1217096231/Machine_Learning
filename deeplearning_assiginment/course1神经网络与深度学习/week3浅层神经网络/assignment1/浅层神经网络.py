#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2

import numpy as np
import matplotlib.pyplot as plt
import random
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import sklearn
import sklearn.datasets
import sklearn.linear_model

np.random.seed(1)   # 设置随机数种子

"""
testCases：提供测试样本，使你能够获得函数的正确性
planar_utils: 提供作业中需要用到的各种函数
X训练集是含有2个特征的400个训练样本的集合；Y是对应的标签
"""
# 加载数据集
X, Y = load_planar_dataset()
print(X.shape)
print(Y.shape)
# 可视化数据集
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
plt.show()
"""获得训练集的信息"""
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]
print("训练集的X shape:", shape_X)
print("训练集的Y shape:", shape_Y)
print("训练样本个数: {}".format(m))
"""简单逻辑回归模型"""
# 下面使用的是scikit-learn实现方式
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
# 显示决策面
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
# 打印精度
Y_hat = clf.predict(X.T)
print("逻辑回归的精度: {}%".format(float(np.dot(Y, Y_hat) + np.dot(1-Y, 1-Y_hat)) / float(Y.size) * 100))
"""下面使用神经网络模型"""


# 定义神经网络结构
def layer_sizes(X, Y):
    """
    返回浅层神经网络的每个层的神经元个数
    :param X:输入的训练集数据----(特征数,样本数)
    :param Y:输出的训练集标签----(标签值,样本数)
    :return:每层神经网络含有的神经元个数
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y

# 测试上述函数
X_assess, Y_assess = layer_sizes_test_case()
n_x, n_h, n_y = layer_sizes(X_assess, Y_assess)
print("输入层神经元个数: ", n_x)
print("隐藏层神经元个数: ", n_h)
print("输出层神经元个数: ", n_y)


# 模型参数初始化
def initialize_parameters(n_x, n_h, n_y):
    """
    初始化神经网络的参数
    :param n_x: 输入层神经元个数
    :param n_h: 隐藏层神经元个数
    :param n_y: 输出层神经元个数
    :return:
    """
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    # 检查变量维度
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params
# 测试上述函数
n_x, n_h, n_y = initialize_parameters_test_case()
params = initialize_parameters(n_x, n_h, n_y)
print("W1 = ", params["W1"])
print("b1 = ", params["b1"])
print("W2 = ", params["W2"])
print("b2 = ", params["b2"])


# 前向传播
def forward_propagation(X, params):
    """
    实现前向传播
    :param X: 输入的训练样本
    :param params: 上一个函数得到的权重和偏执
    :return: 预测值Y_hat
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    Y_hat = sigmoid(Z2)
    assert (Y_hat.shape == (1, X.shape[1]))
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "Y_hat": Y_hat}
    return Y_hat, cache

# 测试上述函数
X_assess, params = forward_propagation_test_case()
Y_hat, cache = forward_propagation(X_assess, params)
print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['Y_hat']))


# 计算代价函数
def compute_cost(Y_hat, Y, params):
    """
    计算代价函数的值
    :param Y_hat: 预测值
    :param Y: 真实值
    :param params:
    :return: cost的值
    """
    m = Y.shape[1]
    logprob = np.multiply(np.log(Y_hat), Y) + np.multiply((1-Y), np.log(1-Y_hat))
    cost = -1 / m * np.sum(logprob)
    cost = np.squeeze(cost)
    assert (isinstance(cost, float))
    return cost
# 测试上述函数
Y_hat, Y_assess, params = compute_cost_test_case()
print("cost = ", compute_cost(Y_hat, Y_assess, params))


# 反向传播
def backward_propagation(params, cache, X, Y):
    """
    计算反向传播
    :param params:
    :param cache:
    :param X:
    :param Y:
    :return:
    """
    m = X.shape[1]
    W1 = params["W1"]
    W2 = params["W2"]
    A1 = cache["A1"]
    Y_hat = cache["Y_hat"]
    dZ2 = Y_hat - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dZ2": dZ2, "dW2": dW2, "db2": db2, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return grads
# 测试上述函数的值
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print("dW1 = ", grads["dW1"])
print("db1 = ", grads["db1"])
print("dW2 = ", grads["dW2"])
print("db2 = ", grads["db2"])


# 更新参数
def update_parameters(params, grads, learning_rate=1.2):
    """
    使用梯度下降法更新参数
    :param params:
    :param grads:
    :param learning_date:
    :return:
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    # 更新参数
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 = learning_rate * db2
    updated_params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return updated_params
# 测试上述函数
params, grads = update_parameters_test_case()
updated_params = update_parameters(params, grads)
print("W1 = ", updated_params["W1"])
print("b1 = ", updated_params["b1"])
print("W2 = ", updated_params["W2"])
print("b2 = ", updated_params["b2"])


# 综合上面的所有函数，搭建一个完整的模型
def nn_model(X, Y, n_h, num_epochs=10000, print_cost=True):
    """

    :param X:
    :param Y:
    :param n_h:
    :param num_epochs: 梯度下降次数
    :param print_cost: 每1000次输出一次loss值
    :return:
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    # 初始化神经网络参数
    params = initialize_parameters(n_x, n_h, n_y)
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    # 梯度下降更新参数
    for i in range(num_epochs):
        # 前向传播
        Y_hat, cache = forward_propagation(X, params)
        # 计算cost值
        cost = compute_cost(Y_hat, Y, params)
        # 反向传播
        grads = backward_propagation(params, cache, X, Y)
        # 更新参数
        updated_params = update_parameters(params, grads)
        if print_cost and i % 100 == 0:
            print("经过梯度下降%d次后的，cost值是:%.4f" % (i, cost))
    return updated_params
# 测试上述函数
X_assess, Y_assess = nn_model_test_case()
updated_params = nn_model(X_assess, Y_assess, 4, num_epochs=1000, print_cost=True)
print("W1 = ", updated_params["W1"])
print("b1 = ", updated_params["b1"])
print("W2 = ", updated_params["W2"])
print("b2 = ", updated_params["b2"])


# 预测函数
def predict(params, X):
    Y_hat, cache = forward_propagation(X, params)
    Y_pred = np.array([1 if x > 0.5 else 0 for x in Y_hat.reshape(-1, 1)]).reshape(Y_hat.shape)
    return Y_pred

# 列表生成式的作用
a = np.array([[0, 2, 1, 3, 5, 7, 6, 5]])
b = np.array([1 if x > 3 else 0 for x in a.reshape(-1, 1)])
b = b.reshape(a.shape)
print(b)
# 测试上述函数功能
params, X_assess = predict_test_case()
Y_pred = predict(params, X_assess)
print("预测值: ", Y_pred)
print("预测平均值: ", np.mean(Y_pred))
# 在planar 数据集上运行nn_model
params = nn_model(X, Y, n_h=4, num_epochs=10000, print_cost=True)
# 绘制决策面
plot_decision_boundary(lambda x: predict(params, x.T), X, Y)
plt.title("hidden layer size is 4")
plt.show()
# 打印出精度
Y_pred = predict(params, X)
print("浅层神经网络的精度: %.2f%%" % float((np.dot(Y, Y_pred.T) + np.dot(1-Y, 1-Y_pred.T))/float(Y.size)*100))
# 调整隐藏层神经元的个数
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('hidden layer unit is %d' % n_h)
    params = nn_model(X, Y, n_h, num_epochs=5000)
    plot_decision_boundary(lambda x: predict(params, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, Y_pred.T) + np.dot(1-Y, 1-Y_pred.T))/float(Y.size)*100)
    print("隐藏层神经元个数是%d时，模型的精度是: %.2f%%" % (n_h, accuracy))
    print("隐藏层神经元是%d的网络已经训练完成!" % n_h)
plt.show()
# 可视化化其他数据集
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
dataset = "gaussian_quantiles"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
if dataset == "blobs":
    Y = Y % 2
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
plt.show()

# 将训练好的模型应用到其他数据集上
i = 0
plt.figure(figsize=(8, 16))
for dataset in datasets:
    plt.subplot(4, 2, i+1)
    i += 1
    plt.title(dataset)
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    if dataset == "blobs":
        Y = Y % 2
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
    params = nn_model(X, Y, n_h=4, num_epochs=10000, print_cost=False)
    plt.subplot(4, 2, i + 1)
    i += 1
    plot_decision_boundary(lambda x: predict(params, x.T), X, Y)
    plt.title(dataset + 'Classifier')
plt.show()