#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-05 10:21:49
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

"""
以在线社区留言为例。为了不影响社区的发展，我们要屏蔽侮辱性的言论，所以要构建一个快速过滤器，
如果某条留言使用了负面或者侮辱性的语言，那么就将该留言标志为内容不当。
对此问题建立两个类型：侮辱类和非侮辱类，使用1和0分别表示。

我们把文本看成单词向量或者词条向量，也就是说将句子转换为向量。
考虑出现所有文档中的单词，再决定将哪些单词纳入词汇表或者说所要的词汇集合，
然后必须要将每一篇文档转换为词汇表上的向量。简单起见，我们先假设已经将本文切分完毕，存放到列表中，并对词汇向量进行分类标注
"""
import numpy as np
from functools import reduce

# 加载每一篇文档，然后将每一篇文档(每个人的留言)经过切分，得到每个列表


def loadDateSet():
    primaryVocabularyLists = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                              ['maybe', 'not', 'take', 'him',
                                  'to', 'dog', 'park', 'stupid'],
                              ['my', 'dalmation', 'is', 'so',
                               'cute', 'I', 'love', 'him'],
                              ['stop', 'posting', 'stupid',
                                  'worthless', 'garbage'],
                              ['mr', 'licks', 'ate', 'my', 'steak',
                               'how', 'to', 'stop', 'him'],
                              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]     # 每篇文档所属的类别，1代表是侮辱性词汇，0代表不是侮辱性词汇
    return primaryVocabularyLists, classVec
# 将每篇文档中的单词提取出来组成一个集合(即每个单词只出现一次)，此集合称为词汇表


def createVocabList(dataSet):  # dataSet --- 每篇文档或每条留言切分后得到的每个列表组成的样本数据集
    vocabuSet = set([])  # 创建一个空的集合，用于存放所有留言中出现的单词
    for document in dataSet:  # 遍历每篇文档/每条留言切分后得到的每个列表
        vocabuSet = vocabuSet | set(document)  # 取并集
    return list(vocabuSet)  # vocabuSet -----存放的是：所有留言中出现的单词(每个单词只出现一次)


# vocabulist即vocabuSet,inputSet即一篇新的文档或者一条新的留言
# 如果inputSet中的单词出现在vocabuList中，则记为1，反之记为0
# 返回的结果returnVec是将每篇文档/每条留言中的单词数字化，转化为词向量


def setOfWords2Vec(vocabuList, inputSet):
    returnVec = [0] * len(vocabuList)  # 创建一个与vocabuList相同长度的全0的向量
    for word in inputSet:   # 遍历每条留言/每篇文档中的每个单词
        if word in vocabuList:    # 如果每篇文档/每条留言中的单词出现在vocabuList中，则returnVec将对应于vocabuList索引位置上的元素置1，反之未出现时置0
            returnVec[vocabuList.index(word)] = 1
        else:
            print('单词:%s不在词汇表中！' % word)
    return returnVec  # returnVec是将每篇文档/每条留言中的单词数字化，转化为词向量

# 训练算法函数
# trainMatrix ---- 每篇文档/每条留言中的单词转化为数字后得到的词向量(returnVec)组成的训练矩阵trainMatrix
# trainCategory ----- 每篇文档所属的类别，即loadDataSet返回的类别标签向量classVec


def trainNB0(trainMatrix, trainCategory):
    # 训练矩阵的行数即含有的训练文档总数
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])                             # 计算每篇文档含有的单词个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)        # 文档属于侮辱类别的概率p(1)
    # 创建numpy.zeros数组,存放每个单词出现的次数
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类单词的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类单词的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    print('属于侮辱性类别中每个单词出现的总次数:', p1Num)
    print('属于侮辱性类别总的单词数目:', p1Denom)
    print('属于非侮辱性类别中每个单词出现的总次数:', p0Num)
    print('属于非侮辱性类别总的单词数目:', p0Denom)
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive
# 朴素贝叶斯分类函数


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1       # 对应元素相乘
    p0 = reduce(lambda x, y: x * y, vec2Classify * p0Vec) * (1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0
# 测试算法


def testingNB():
    primaryVocabularyLists, classVec = loadDateSet()
    print('primaryVocabularyLists:\n', primaryVocabularyLists)
    myVocabuList = createVocabList(primaryVocabularyLists)  # 创建词汇表，每个单词只出现一次
    print('myVocabuList:\n', myVocabuList)   # 打印输出一下词汇表
    trainMat = []  # trainMat是将每篇文档或每条留言转化为数字0，1后的结果即returnVec，主要利用的是SetOfWords2Vec()函数来实现
    for primaryVocabularyList in primaryVocabularyLists:
        trainMat.append(setOfWords2Vec(myVocabuList, primaryVocabularyList))
    print('单词转化为数字后的训练矩阵trainMat:\n', trainMat)
    print('需要训练的总文档数目:', len(trainMat))  # trainMat是由0，1组成的多维数组，称为训练矩阵
    print('词汇表中的总词条数:', len(trainMat[0]))
    p0V, p1V, pAb = trainNB0(trainMat, classVec)   # 开始训练分类函数
    print('属于非侮辱类单词的条件概率数组p0V:\n', p0V)
    print('属于侮辱类单词的条件概率数组p1V:\n', p1V)
    print('类别标签向量classVec:\n', classVec)
    print('文档属于侮辱类文档的概率pAb:\n', pAb)
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabuList, testEntry))  # 测试样本向量化
    print('thisDoc测试样本1向量化后转化为0,1数字矩阵的结果：', thisDoc)
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果

    testEntry = ['stupid', 'garbage']  # 测试样本2
    thisDoc = np.array(setOfWords2Vec(myVocabuList, testEntry))  # 测试样本向量化
    print('thisDoc测试样本2向量化后转化为0,1数字矩阵的结果：', thisDoc)
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')     # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果


if __name__ == '__main__':
    testingNB()
