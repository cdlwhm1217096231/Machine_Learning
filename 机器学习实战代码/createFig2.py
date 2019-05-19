#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-19 16:09:07
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import PCA

dataMat = PCA.loadDataSet(
    'E:\\Code\\local_code\\机器学习实战\\DataSet\\testSetPCA.txt')
lowDMat, reconMat = PCA.pca(dataMat, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:, 0].flatten().A[0],
           dataMat[:, 1].flatten().A[0], marker='^', s=60)
ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:,
                                                   1].flatten().A[0], marker='o', s=50, c='red')
plt.show()
