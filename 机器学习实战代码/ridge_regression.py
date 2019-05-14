#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-14 08:25:57
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from bs4 import BeautifulSoup
import random

# 加载数据集


def loadDataSet(fileName):
    num_features = len(open(fileName).readline().split('\t')) - 1
    xArry = []
    yArry = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArry = []
        curLine = line.strip().split("\t")
        for i in range(num_features):
            lineArry.append(float(curLine[i]))
        xArry.append(lineArry)
        yArry.append(float(curLine[-1]))
    return xArry, yArry


# 岭回归
def ridge_regression(xMat, yMat, lamda=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lamda
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵，不可逆")
    ws = denom.I * (xMat.T * yMat)
    return ws

# 岭回归的测试函数


def ridge_regression_test(xArry, yArry):
    xMat = np.mat(xArry)
    yMat = np.mat(yArry).T
    # 数据标准化
    yMean = np.mean(yMat, axis=0)  # 行与行操作，求均值
    yMat = yMat - yMean  # 数据减去均值
    xMean = np.mean(xMat, axis=0)  # 行与行操作，求均值
    xVar = np.var(xMat, axis=0)  # 行与行操作，求方差
    xMat = (xMat - xMean) / xVar                        # 数据减去均值除以方差来实现标准化
    numTestPts = 30  # 30个不同的lambda测试
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))  # 初始化回归系数矩阵
    for i in range(numTestPts):   # 改变lambda计算回归系数
        # lambda以e的指数变化，最初是一个非常小的数,计算回归系数矩阵
        ws = ridge_regression(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


# 绘制岭回归系数矩阵
def plotwMat():
    font = FontProperties(fname=r"C:\\Windows\Fonts\\simsun.ttc", size=14)
    abX, abY = loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\abalone.txt")
    ridgeWeights = ridge_regression_test(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    ax_title_text = ax.set_title(
        u'log(lambada)与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


def rssError(yArry, yHat):
    return ((yArry - yHat)**2).sum()


# 数据标准化
def stanrandlize(xMat, yMat):
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    yMean = np.mean(yMat, 0)
    inyMat = yMat - yMean
    inMeans = np.mean(inxMat, 0)
    inVar = np.var(inxMat, 0)
    inxMat = (inxMat - inMeans) / inVar
    return inxMat, inyMat


# 前向逐步线性回归
def stageWise(xArry, yArry, eps=0.01, epoch=100):
    xMat = np.mat(xArry)
    yMat = np.mat(yArry).T
    xMat, yMat = stanrandlize(xMat, yMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((epoch, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(epoch):
        lowestError = float("inf")
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# 绘制前向逐步回归的回归系数矩阵
def plotstageWiseMat():
    font = FontProperties(fname=r"C:\\Windows\Fonts\\simsun.ttc", size=14)
    xArry, yArry = loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\abalone.txt")
    returnMat = stageWise(xArry, yArry, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


# 利用线性回归预测乐高玩具套装的价格
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    函数说明:从页面读取数据，生成retX和retY列表
    Parameters:
        retX - 数据X
        retY - 数据Y
        inFile - HTML文件
        yr - 年份
        numPce - 乐高部件数目
        origPrc - 原价
    Returns:
        无
    """
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while(len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" %
                      (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)


def setDataCollect(retX, retY):
    """
    函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
    Parameters:
        无
    Returns:
        无
    """
    scrapePage(retX, retY, 'E:\\Code\\local_code\\机器学习实战\\DataSet\\lego\\lego8288.html',
               2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, 'E:\\Code\\local_code\\机器学习实战\\DataSet\\lego\\lego10030.html',
               2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, 'E:\\Code\\local_code\\机器学习实战\\DataSet\\lego\\lego10179.html',
               2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, 'E:\\Code\\local_code\\机器学习实战\\DataSet\\lego\\lego10181.html',
               2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, 'E:\\Code\\local_code\\机器学习实战\\DataSet\\lego\\lego10189.html',
               2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, 'E:\\Code\\local_code\\机器学习实战\\DataSet\\lego\\lego10196.html',
               2009, 3263, 249.99)  # 2009年的乐高10196,部件数目3263,原价249.99


# 标准线性回归
def standRegression(xArry, yArry):
    xMat = np.mat(xArry)
    yMat = np.mat(yArry).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不可逆")
    ws = xTx * (xMat.T * yMat)
    return ws


# 使用标准线性回归
def useStandRegression():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    data_num, num_features = np.shape(lgX)
    lgX1 = np.mat(np.ones((data_num, num_features + 1)))
    lgX1[:, 1:5] = np.mat(lgX)
    ws = standRegression(lgX1, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' %
          (ws[0], ws[1], ws[2], ws[3], ws[4]))



# 交叉验证
def crossValidation(xArr, yArr, numVal=10):
    """
    函数说明:交叉验证岭回归
    Parameters:
        xArr - x数据集
        yArr - y数据集
        numVal - 交叉验证次数
    Returns:
        wMat - 回归系数矩阵
    """
    m = len(yArr)                                   # 统计样本个数
    indexList = list(range(m))                      # 生成索引值列表
    errorMat = np.zeros((numVal,30))           # create error mat 30columns numVal rows
    for i in range(numVal):                              # 交叉验证numVal次
        trainX = []
        trainY = []                                # 训练集
        testX = []
        testY = []                                 # 测试集
        random.shuffle(indexList)                  # 打乱次序
        for j in range(m):                         # 划分数据集:90%训练集，10%测试集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridge_regression_test(trainX, trainY)                #  获得30个不同lambda下的岭回归系数
        for k in range(30):                             #  遍历所有的岭回归系数
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)           # 测试集
            meanTrain = np.mean(matTrainX,0)                               # 测试集均值
            varTrain = np.var(matTrainX,0)                                 # 测试集方差
            matTestX = (matTestX - meanTrain) / varTrain                   # 测试集标准化
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)        # 根据ws预测y值
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))           # 统计误差
    meanErrors = np.mean(errorMat,0)                                       # 计算每次交叉验证的平均误差
    minMean = float(min(meanErrors))                                       # 找到最小误差
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]             # 找到最佳回归系数
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights / varX                                #  数据经过标准化，因此需要还原
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))


# 使用sklearn中的函数
def use_sklearn():
    from sklearn import linear_model
    reg = linear_model.Ridge(alpha = 0.5)
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    reg.fit(lgX, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3]))


if __name__ == '__main__':
    plotwMat()
    plotstageWiseMat()
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    print(ridge_regression_test(lgX, lgY))
    useStandRegression()
    crossValidation(lgX, lgY)  # 交叉验证
    use_sklearn()
