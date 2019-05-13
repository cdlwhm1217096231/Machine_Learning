#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-13 20:35:59
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集


def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        xArr - x数据集
        yArr - y数据集
    """

    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def plotDataSet():
    """
    函数说明:绘制数据集
    Parameters:
        无
    Returns:
        无
    """
    xArr, yArr = loadDataSet(
        'E:\\Code\\local_code\\机器学习实战\\DataSet\\ex0.txt')  # 加载数据集
    n = len(xArr)  # 数据个数
    xcord = []
    ycord = []  # 样本点
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.ylabel("Y")
    plt.show()


# 标准线性回归
def standRegres(xArr, yArr):
    """
    函数说明:计算回归系数w
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat  # 根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部加权线性回归
def plotlwlrRegression():
    """
    函数说明:绘制多条局部加权回归曲线
    Parameters:
        无
    Returns:
        无
    """
    font = FontProperties(fname=r"C:\\Windows\Fonts\\simsun.ttc", size=14)
    xArr, yArr = loadDataSet('E:\\Code\\local_code\\机器学习实战\\DataSet\\ex0.txt')  # 加载数据集
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)  # 根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)  # 根据局部加权线性回归计算yHat
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)  # 根据局部加权线性回归计算yHat
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    srtInd = xMat[:, 1].argsort(0)  # 排序，返回索引值
    xSort = xMat[srtInd][:, 0, :]
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False,
                            sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')  # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')  # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')  # 绘制回归曲线
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0],
                   s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0],
                   s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0],
                   s=20, c='blue', alpha=.5)  # 绘制样本点
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(
        u'局部加权回归曲线,k=0.003', FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

# 局部加权线性回归


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    函数说明:使用局部加权线性回归计算回归系数w
    Parameters:
        testPoint - 测试样本点
        xArr - x数据集
        yArr - y数据集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))  # 创建权重对角矩阵
    for j in range(m):  # 遍历数据集计算每个样本的权重
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))  # 计算回归系数
    return testPoint * ws

# 局部加权线性回归测试


def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    函数说明:局部加权线性回归测试
    Parameters:
        testArr - 测试数据集
        xArr - x数据集
        yArr - y数据集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数
    """
    m = np.shape(testArr)[0]  # 计算测试数据集大小
    yHat = np.zeros(m)
    for i in range(m):  # 对每个样本点进行预测
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def rssError(yArr, yHatArr):
    """
    误差大小评价函数
    Parameters:
        yArr - 真实数据
        yHatArr - 预测数据
    Returns:
        误差大小
    """
    return ((yArr - yHatArr) ** 2).sum()


if __name__ == '__main__':
    plotDataSet()
    xArr, yArr = loadDataSet(
        'E:\\Code\\local_code\\机器学习实战\\DataSet\\ex0.txt')  # 加载数据集
    ws = standRegres(xArr, yArr)  # 计算回归系数
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    yHat = xMat * ws
    print(np.corrcoef(yHat.T, yMat))
    plotlwlrRegression()  # 绘制局部加权线性回归曲线
    # 预测鲍鱼年龄
    abX, abY = loadDataSet(
        'E:\\Code\\local_code\\机器学习实战\\DataSet\\abalone.txt')
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[0:99], yHat10.T))
    print('')
    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[100:199], yHat10.T))
    print('')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))
