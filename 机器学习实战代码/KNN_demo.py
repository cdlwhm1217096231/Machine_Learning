#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
K近邻算法：监督学习的分类算法
'''
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator

"""
classify0(inX, dataSet, labels, k)函数说明:kNN分类器函数------------------------约会网站配对效果判定----------------------------------

参数说明:
inX - 用于分类的数据(测试集)
dataSet - 用于训练的数据(训练集)
labels - 分类标签
k - kNN算法参数,选择距离最小的k个点
返回值- sortedClassCount[0][0] - 分类结果
"""


# 分类器函数
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # numpy函数shape[0]返回dataSet的行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # # inX在列方向上重复共1次(横向),在行方向上重复共dataSetSize次
    sqDiffMat = diffMat**2  # 特征相减后进行平方
    sqDistances = sqDiffMat.sum(axis=1)  # 列元素压缩为一列，具体压缩方法是各列元素相加，最终得到一列
    distances = sqDistances**0.5  # 开方得欧式距离
    sortedDistIndices = distances.argsort()  # 返回distances中元素从小到大排序后的索引值
    classCount = {}  # 定义一个记录类别次数的字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]  # 取出前k个元素的类别
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算类别次数
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # reverse降序排序字典
        return sortedClassCount[0][0]  # 返回次数最多的类别,即所要分类的类别
"""
file2matrix(filename)函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力
参数说明:
	filename - 文件名
返回值：
	returnMat - 特征矩阵
	classLabelVector - 分类Label向量
"""


def file2matrix(filename):
    fr = open(filename)   # 打开文件
    arrayOLines = fr.readlines()  # 逐行读取文件所有内容
    numberOfLines = len(arrayOLines)  # 得到文件行数
    returnMat = np.zeros((numberOfLines, 3))  # 返回的零矩阵：numberOfLines行,3列
    classLabelVector = []  # 返回的分类标签向量
    index = 0   # 行的索引值
    for lines in arrayOLines:
        line = lines.strip()
        listFromLine = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector
"""
函数说明:可视化数据

参数:
	datingDataMat - 特征矩阵
	datingLabels - 分类Label
无返回值
"""


def showdatas(datingDataMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()
"""
autoNorm(dataSet)函数说明:对数据进行归一化

参数:
	dataSet - 特征矩阵
返回值:
	normDataSet - 归一化后的特征矩阵
	ranges - 数据范围
	minVals - 数据最小值
"""


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
"""
datingClassTest()函数说明:分类器测试函数
参数:
	无
返回值:
	normDataSet - 归一化后的特征矩阵
	ranges - 数据范围
	minVals - 数据最小值
"""


def datingClassTest():
    filename = "datingTestSet.txt"  # 打开的文件名
    datingDataMat, datingLabels = file2matrix(filename)  # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    hoRatio = 0.10  # 取所有数据的百分之十
    normMat, ranges, minVals = autoNorm(datingDataMat) # # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    m = normMat.shape[0]  # 获得normMat的行数
    numTestVecs = int(m * hoRatio)  # 百分之十的测试数据的个数
    errorCount = 0.0  # 分类错误计数器
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)   # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))  # 格式化输出
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f" % (errorCount / float(numTestVecs) * 100))
"""
classifyPerson()函数说明:通过输入一个人的三维特征,进行分类输出
参数:
    无
返回值:
    无
"""


def classifyPerson():
    resultList = ['讨厌', '有些喜欢', '非常喜欢']  # 输出结果
    precentTats = float(input("玩视频游戏所耗时间百分比:"))  # 三维特征用户输入
    ffMiles = float(input("每年获得的飞行常客里程数:"))    # 三维特征用户输入
    iceCream = float(input("每周消费的冰激淋公升数:"))      # 三维特征用户输入
    filename = "datingTestSet.txt"  # 打开的文件名
    datingDataMat, datingLabels = file2matrix(filename)  # 打开并处理数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 训练集归一化
    inArr = np.array([precentTats, ffMiles, iceCream])  # 生成NumPy数组,测试集
    norminArr = (inArr - minVals) / ranges              # 测试集归一化
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)   # 返回分类结果
    print("你可能%s这个人" % (resultList[classifierResult]))      # 打印结果


"""
函数说明:main函数
"""
if __name__ == '__main__':
    # 1、测试代码
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    # 2、测试代码，得到错误率
    datingClassTest()
    # 3、使用算法
    classifyPerson()






















