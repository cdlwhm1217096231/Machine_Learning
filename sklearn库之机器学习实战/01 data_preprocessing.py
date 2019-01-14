#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-14 18:38:55
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import pandas as pd


dataset = pd.read_csv("./datasets/Data.csv")
print(dataset.head())
print(dataset.describe())
print("*" * 30)
# step1: importing dataset
"""
loc：通过行标签索引数据

iloc：通过行号索引行数据

ix：通过行标签或行号索引数据（基于loc和iloc的混合）
"""
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print("step1: importing dataset")
print("X=\n", X)
print("Y=\n", Y)
# step2: handling the missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)  # 使用均值填充缺失值
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("*" * 30)
print("step2: handling the missing data")
print("X=\n", X)
print("Y=\n", Y)
# step3: Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Labelencoder_X = LabelEncoder()
X[:, 0] = Labelencoder_X.fit_transform(X[:, 0])  # 将文本属性转化为数字属性
# creating  a dummy variable
Onehotencoder = OneHotEncoder(categorical_features=[0])
X = Onehotencoder.fit_transform(X).toarray()
Labelencoder_Y = LabelEncoder()
Y = Labelencoder_Y.fit_transform(Y)
print("*" * 30)
print("step3: Encoding categorical data")
print("X=\n", X)
print("Y=\n", Y)
# step4: splitting the datasets into training sets and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
print("*" * 30)
print("step4: splitting the datasets into training sets and test sets")
print("X_train:\n", X_train)
print("Y_train:\n", Y_train)
print("X_test:", X_test)
print("Y_test:\n", Y_test)
# step5: feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("*" * 30)
print("step5: feature scaling")
print("X_train:\n", X_train)
print("Y_test:\n", Y_test)
