#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import ndimage
from PIL import Image
from lr_utils import load_dataset


# 备注函数np.squeeze()的作用
a = np.random.randn(2, 3, 1)
print(a, a.shape)
b = np.squeeze(a)
print(b, b.shape)

# 加载原始数据集---增加orig是因为一会还要进行预处理，预处理之后使用不带orig的后缀
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
# 可视化图片
index = 5
plt.imshow(train_set_x_orig[index])
plt.show()
print("y = " + str(train_set_y_orig[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y_orig[:, index])].decode("utf-8") + "' picture.")
"""
m_train:训练样本的数量
m_test: 测试样本的数量
num_px: 训练集中图片的高度与宽度,此处：图片的宽度=图片的高度
train_set_x_orig的形状是：(m_train, num_px, num_px, 3)
"""
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print("训练样数量:", m_train)
print("测试样本数量:", m_test)
print("图片的宽度或高度:", num_px)
print("每张图片的尺寸: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("训练集x的shape:", train_set_x_orig.shape)
print("训练集y的shape:", train_set_y_orig.shape)
print("测试集x的shape:", test_set_x_orig.shape)
print("测试集y的shape:", test_set_y_orig.shape)
"""
图片预处理:转化图片的形状（num_px, num_px, 3）--->  (num_px * num_px * 3, 1)
"""
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # -1表示转置前是任意列数量，转置后是每列代表一个样本
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print("flatten后训练集x的shape:", train_set_x_flatten.shape)
print("训练集y的shape:", train_set_y_orig.shape)
print("flatten后测试集x的shape:", test_set_x_flatten.shape)
print("测试集y的shape:", test_set_y_orig.shape)
print("确认变换后的形状:", train_set_x_flatten[0:5, 0])  # 第一个样本的，前5个特征
# 对flatten后的数据进行标准化与中心化，即特征缩放，使得像素点在0-255之间的像素值变为0到1之间的值
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# 辅助函数
def sigmiod(z):
    """
    激活函数sigmoid
    :param z: 标量或者向量
    :return: sigmoid函数的值
    """
    s = 1 / (1 + np.exp(-z))
    return s

# 测试上述函数的结果
print("sigmoid([0, 2]) = ", str(sigmiod(np.array([0, 2]))))


# 参数初始化函数
def initialize_with_zeros(dim):
    """
    初始化参数W函数
    :param dim: 权重w的维度
    :return: 已经初始化的权重W----(dim, 1);b是一个标量
    """
    W = np.zeros((dim, 1))
    b = 0
    assert (W.shape == (dim, 1))   # 此处加上一个assert来声明变量的维度，如果是False，说明此处声明时错误的
    assert (isinstance(b, float) or isinstance(b, int))
    return W, b

# 测试上述函数
dim = 2
W, b = initialize_with_zeros(dim)
print("W ={}".format(W))
print("b ={}".format(b))
"""
前向传播与反向传播
"""


def propagate(W, b, X, Y):
    """
    前向传播与反向传播函数
    :param W: 权重---->  （num_px*num_px*3, 1)
    :param b: 偏置，一个标量
    :param X: 输入，---->(num_px*num_px*3, 样本数)
    :param Y: 输出,0或者1
    :return: cost,dW, db
    """
    m = X.shape[1]
    # 前向传播
    A = sigmiod(np.dot(W.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    # 反向传播
    dW = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    # 参数维度检查
    assert (dW.shape == W.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)  # 从数组的形状中删除为1的维度
    assert (cost.shape == ())

    grads = {"dW": dW, "db": db}
    return grads, cost
# 测试上述函数
W, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
grads, cost = propagate(W, b, X, Y)
print("dW = {}".format(grads["dW"]))
print("db = {}".format(grads["db"]))
print("cost = {}".format(cost))


# 优化过程
def optimiz(W, b, X, Y, num_epochs, learning_rate, print_cost=False):
    """
    通过梯度下降算法，优化W，b，找到最优的W, b
    :param W: 权重---->  （num_px*num_px*3, 1)
    :param b: 偏置，一个标量
    :param X: 输入，---->(num_px*num_px*3, 样本数)
    :param Y: 输出,0或者1
    :param num_epochs: 梯度下降的次数
    :param learning_rate: 学习率
    :param print_cost: 没100次梯度下降后，打印出loss值
    :return: W，b的参数，梯度， costs的值
    """
    costs = []
    for i in range(num_epochs):
        grads, cost = propagate(W, b, X, Y)
        dW = grads["dW"]
        db = grads["db"]
        # 更新参数
        W = W - learning_rate * dW
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("经过%d次梯度下降后,cost的值是%.2f" % (i, cost))
    params = {"W": W, "b": b}
    grads = {"dW": dW, "db": db}
    return params, grads, costs

# 测试上述函数
params, grads, costs = optimiz(W, b, X, Y, num_epochs=100, learning_rate=0.009, print_cost=False)
print("W = {}".format(params["W"]))
print("b = {}".format(params["b"]))
print("dW = {}".format(grads["dW"]))
print("db = {}".format(grads["db"]))
print(costs)
"""
预测函数:使用已经优化后的W，b预测未知样本
"""


def predict(W, b, X):
    """
    使用已经优化后的W，b去预测输入的X的标签Y是0还1
    :param W: 权重--->（num_px * num_px * 3, 1）
    :param b: 偏置，标量
    :param X: 训练样本(num_px * num_px * 3, 样本数m)
    :return: 预测值y_pred
    """
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    W = W.reshape(X.shape[0], 1)
    A = sigmiod(np.dot(W.T, X) + b)
    for i in range(X.shape[1]):
        if A[0, i] <= 0.5:
            Y_pred[0, i] = 0
        else:
            Y_pred[0, i] = 1
    assert (Y_pred.shape == (1, m))
    return Y_pred
# 测试上面的函数
print("predictions={}".format(predict(W, b, X)))
"""
组合上面的各种函数，成为一个完整的模型
"""


def model(X_train, Y_train, X_test, Y_test, num_epochs=2000, learning_rate=0.5, print_cost=False):
    """
    构建逻辑回归模型
    :param X_train: 训练集数据X--->（num_px * num_px * 3, 训练样本数）
    :param Y_train: 训练集标签Y ---> (1, 训练样本数)
    :param X_test: 测试集数据X--->（num_px * num_px * 3, 测试样本数）
    :param Y_test: 测试集标签Y ---> (1, 测试样本数)
    :param num_epochs: 梯度下降次数
    :param learning_rate: 学习率
    :param print_cost: 每100次，打印一次cost
    :return: 包含模型的信息字典d
    """
    # 初始化参数
    W, b = initialize_with_zeros(X_train.shape[0])
    # 梯度下降
    paramters, grads, costs = optimiz(W, b, X_train, Y_train, num_epochs, learning_rate, print_cost)
    # 从parameters字典中检索最优参数
    W = paramters["W"]
    b = paramters["b"]
    # 预测
    Y_pred_test = predict(W, b, X_test)
    Y_pred_train = predict(W, b, X_train)
    # 打印在训练集合测试集上的精度
    print("训练集上的精度:{} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("测试集上的精度:{} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_pred_train": Y_pred_train,
         "Y_pred_test": Y_pred_test,
         "W": W,
         "b": b,
         "学习率": learning_rate,
         "梯度下降次数": num_epochs,
         }
    return d
# 测试上述函数model
d = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_epochs=2000, learning_rate=0.005, print_cost=True)
# 上面的model已经过拟合了，在测试集上进行测试
index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
plt.show()
print("y = " + str(test_set_y_orig[0, index]) + ", 真实值 '" + classes[int(d["Y_pred_test"][0, index])].decode("utf-8") + " '图片")
# 绘制cost曲线
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.xlabel("num_epochs")
plt.ylabel("cost")
plt.title("Learning rate = " + str(d["学习率"]))
plt.show()
# 调整学习率
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for lr in learning_rates:
    print("现在的学习率是:{}".format(lr))
    models[str(lr)] = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_epochs=2000, learning_rate=lr, print_cost=False)
    print("==============================================================")
for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["学习率"]))
plt.xlabel("epochs")
plt.ylabel("cost")
legend = plt.legend(loc="upper center", shadow=True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
plt.show()
# 使用自己的图片进行测试
img = "my_image.jpg"
file_name = "./images/" + img
image = np.array(ndimage.imread(file_name, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["W"], d["b"], my_image)
plt.imshow(image)
plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(image))].decode("utf-8") + "\" picture.")

