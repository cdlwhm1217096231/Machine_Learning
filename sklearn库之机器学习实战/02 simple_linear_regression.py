#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-14 19:00:45
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("./datasets/studentscores.csv")
print(dataset.head())
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 4, random_state=0)

# fitting simple linear regression model to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
# predicting the result
Y_pred = regressor.predict(X_test)
# visualizing the training sets
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.xlabel("X_train")
plt.ylabel("Y_train")
plt.title("visualizing the training sets")
plt.show()

# visualizing the test sets
plt.scatter(X_test, Y_test, color="yellow")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.xlabel("X_test")
plt.ylabel("Y_test")
plt.title("visualizing the test sets")
plt.show()
