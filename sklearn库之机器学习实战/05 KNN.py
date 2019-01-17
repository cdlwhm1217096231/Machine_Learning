#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-17 09:01:48
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("./datasets/Social_Network_Ads.csv")
print(dataset.head())
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# 特征缩放
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 使用KNN模型
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(
    n_neighbors=5, metric="minkowski", p=2)  # 使用的是闵式距离，当p=2时，等价于欧式矩阵

classifier.fit(X_train, Y_train)

# 预测输出
Y_pred = classifier.predict(X_test)

# 混淆矩阵

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


cm = confusion_matrix(Y_test, Y_pred)
print("混淆矩阵:\n", cm)
print("分类报告:\n", classification_report(Y_test, Y_pred))
