#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-19 16:00:32
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from numpy import *
import matplotlib
import matplotlib.pyplot as plt


n = 1000

xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []

markers = []
colors = []
fw = open("E:\\Code\\local_code\\机器学习实战\\DataSet\\testSetPCA.txt", "w")
for i in range(n):
    [r0, r1] = random.standard_normal(2)
    fFlyer = r0 + 9.0
    tats = 1.0 * r1 + fFlyer + 0
    xcord0.append(fFlyer)
    ycord0.append(tats)
    fw.write("%f\t%f\n" % (fFlyer, tats))
fw.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0, ycord0, marker="^", s=90)
plt.xlabel("hours of direct sunlight")
plt.ylabel("liters of water")
plt.show()
