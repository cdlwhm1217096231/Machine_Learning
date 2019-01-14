#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-25 12:31:33
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import tensorflow as tf


# 代价函数cost = w**2 - 10w + 25

coefficients = np.array([[1.0], [-10.], [25.]])
w = tf.Variable(0, dtype=tf.float32)  # w是我们要进行优化的参数，因此称为变量
x = tf.placeholder(tf.float32, [3, 1])  # 将训练集数据x加载到tensorflow中[1, -10, 25].T
# cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)  # 定义的cost代价函数
#  cost = w**2 - 10*w + 25
cost = x[0][0] * w**2 + x[1][0] * w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()  # 初始化全局变量
with tf.Session() as session:
    session.run(init)
    print(session.run(w))  # 尚未开始训练，w=0
    session.run(train, feed_dict={x: coefficients})   # 运行一步梯度下降法
    print('运行梯度下降1次后，再评估w的值：', session.run(w))  # 运行梯度下降1次后，再评估w的值

    for i in range(1000):  # 运行梯度下降1000次迭代
        session.run(train, feed_dict={x: coefficients})
    print('运行梯度下降1000次迭代后，再评估w的值:', session.run(w))  # 运行梯度下降1000次迭代后，再评估w的值
