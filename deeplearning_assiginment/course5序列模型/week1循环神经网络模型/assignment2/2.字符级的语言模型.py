#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2

import numpy as np
import random
from utils import *

f = open('dinos.txt', 'r')
data = f.read()
data = data.lower()
chars = list(set(data))
data_size, chars_size = len(data), len(chars)
print('data中总的大小%d, 不重复的字符有%d个' % (data_size, chars_size))

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}  # 将字符映射成索引
print(char_to_ix)
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}   # 将索引映射成字符
print(ix_to_char)
print("******************************************************************************")

# 梯度剪切-------(防止梯度爆炸)
def clip(gradients, maxValue):
    clipped_gradients = {}
    for g in ['dWax', 'dWaa', 'dWya', 'db', 'dby']:
        clipped_gradients[g] = np.clip(gradients[g], -maxValue, maxValue)
    return clipped_gradients


np.random.seed(3)
dWax = np.random.randn(5, 3) * 10
dWaa = np.random.randn(5, 5) * 10
dWya = np.random.randn(2, 5) * 10
db = np.random.randn(5, 1) * 10
dby = np.random.randn(2, 1) * 10
gradients = {'dWax': dWax, 'dWaa': dWaa, 'dWya': dWya, 'db': db, 'dby': dby}
gradients = clip(gradients, 10)
print("gradients['dWaa'][1][2] =", gradients['dWaa'][1][2])
print("gradients['dWax'][3][1] =", gradients['dWax'][3][1])
print("gradients['dWya'][1][2] =", gradients['dWya'][1][2])
print("gradients['db'][4] =", gradients['db'][4])
print("gradients['dby'][1] =", gradients['dby'][1])
print("******************************************************************************")

# 采样sampling----假设模型已经训练好，使用模型生成下一个字符/文本
def sample(parameters, char_to_ix, seed):
    """
    根据RNN输出的概率分布序列，对字符序列采样
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    # 采样时，初始参数值全部设置为0，即在当前模型的现有参数下，进行恐龙名字第一个字符的预测
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):
        a = np.tanh(np.matmul(Wax, x) + np.matmul(Waa, a_prev) + b)
        z = np.matmul(Wya, a) + by
        y = softmax(z)
        np.random.seed(counter + seed)
        idx = np.random.choice(range(vocab_size), p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        seed += 1
        counter += 1
    if (counter == 50):
        indices.append(char_to_ix['\n'])
    return indices


# 测试 sample(parameters, char_to_ix, seed) 函数
np.random.seed(2)
vocab_size, n_a = 27, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

indices = sample(parameters, char_to_ix, 0)
print("采样:")
print("-- 采样后的字符索引列表:", indices)
print("-- 采样后的字符列表:", [ix_to_char[i] for i in indices])
print("******************************************************************************")


# 梯度下降
def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)
    return loss, gradients, a[len(X)-1]

np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12, 3, 5, 11, 22, 3]
Y = [4, 14, 11, 22, 25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate=0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])
print("******************************************************************************")
# 构建语言模型
# RNN通用优化循环的步骤：
# 1、正向传播通过RNN来计算损失
# 2、反向传播通过时间来计算相对于参数的损失的梯度
# 3、如有必要，梯度裁剪
# 4、使用梯度下降更新您的参数

# 训练集中的一行作为一个样本
# 在有100次随机梯度下降中，每次梯度下降随机选取10个名称来查看该算法的效果
# 每次选取的时候，需要重新洗牌


# 训练模型
def model(data, ix_to_char, char_to_ix, num_iterations=400000, n_a=150, dino_names=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.
    Arguments:
    data -- text corpus（文本语料库）
    ix_to_char -- 将字符索引映射为字符
    char_to_ix -- 将字符映射为字符索引
    num_iterations -- 训练模型的迭代次数
    n_a --  RNN cell中的神经元个数
    dino_names -- 每次迭代时要采样的恐龙名称数量
    vocab_size -- 在文本中找到的唯一字符数量

    Returns:
    parameters -- 学习参数
    """
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))
    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)
        loss = smooth(loss, curr_loss)
        if j % 1000 == 0:
            print('迭代:%d次后, Loss值:%f' % (j, loss))
            seed = 0
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                seed += 1
    return parameters

parameters = model(data, ix_to_char, char_to_ix)