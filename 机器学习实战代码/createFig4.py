#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-19 16:40:13
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import PCA

dataMat = PCA.replaceNanWithMean()

# below is a quick hack copied from pca.pca()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals  # remove mean
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
eigValInd = eigValInd[::-1]  # reverse
sortedEigVals = eigVals[eigValInd]
total = sum(sortedEigVals)
varPercentage = sortedEigVals / total * 100

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^')
plt.xlabel('Principal Component Number')
plt.ylabel('Percentage of Variance')
plt.show()
