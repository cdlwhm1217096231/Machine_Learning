#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-15 10:13:22
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import pandas as pd
import numpy as np

dataset = pd.read_csv("./datasets/50_Startups.csv")
print(dataset.head())

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoder categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(dataset.iloc[:, 3])  # 将第3列的文本属性转化为数字属性

onehotencoder = OneHotEncoder(categorical_features=[3])  # 将第3列转化为独热编码
X = onehotencoder.fit_transform(X).toarray()

# 避免虚拟变量avoiding dummy variable trap
X = X[:, 1:]
# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
# regression evaluation
from sklearn.metrics import r2_score
print("r2_score:", r2_score(Y_test, y_pred))
