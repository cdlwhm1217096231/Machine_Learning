#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-18 11:30:01
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from numpy import *


# 层次聚类，基于查找关联规则Apriori算法


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 构建集合C1,C1是大小为1的所有候选项集的集合，如{{1},{2},{3},{4},{5}}
def createC1(dataSet):
    C1 = []
    for record in dataSet:
        for item in record:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))  # frozenset是对C1进行冰冻，使C1不能进行修改


# 从C1中生成L1(L1是满足最小支持度的要求的项集构成的集合)
def scanD(D, Ck, minSupport):
    """
        D:数据集
        Ck:候选项集列表
        minSupport:感兴趣项集的最小支持度
        返回值：retList---L1;supportData---包含支持度值的字典
    """
    ssCnt = {}  # 创建一个空字典，字典的key就是C1中的集合，value是C1中的集合在所有记录中出现的次数
    for record in D:   # 遍历数据集中的每条记录
        for can in Ck:  # 遍历C1中的所有候选项集
            if can.issubset(record):   # 如果C1中的集合是记录中的一部分，则增加字典中对应的计数值;
                if not can in ssCnt:
                    ssCnt[can] = 1  # 字典的key就是集合
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))  # 总的样本数
    print("总的记录数:", numItems)
    retList = []  # 创建一个空列表，此列表包含满足最小支持度的集合
    supportData = {}  # 最频繁项集的支持度
    for key in ssCnt:  # 遍历字典中的每个元素，并计算其最小支持度
        support = ssCnt[key] / numItems  # 计算支持度
        if support >= minSupport:  # 如果C1中的支持度满足最小支持度的要求，就将字典中的元素加入retList中
            retList.insert(0, key)  # 在列表的首部插入新的集合
        supportData[key] = support  # 最频繁项集的支持度
    return retList, supportData


#
def aprioriGen(Lk, k):   # 创建候选项集Ck，对L1中的元素两两组合，得出候选项集C2
    """
        频繁项集列表LK，项集元素个数K
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])  # 集合的并操作
    return retList


# Apriori核心程序
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData0 = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supportDatak = scanD(D, Ck, minSupport)
        supportData0.update(supportDatak)
        L.append(Lk)
        k += 1
    return L, supportData0


# 关联规则表的生成
def generateRules(L, supportData, minConf=0.7):
    """
        L:频繁项集
        supportData:包含那些频繁项集支持数据的字典
        minConf:最小的可信度
        返回：一个包含可信度的规则列表bigRuleList
    """
    bigRuleList = []  # 初始化存放所有关联规则的列表
    for i in range(1, len(L)):
        for freqSet in L[i]:  # 最开始的关联规则表中，每条关联规则freqSet的右部H1只有一个元素
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 计算可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算每条关联规则的可信度
        print(freqSet - conseq, '--->', conseq, '可信度conf:', conf)
        brl.append((freqSet - conseq, conseq, conf))
        prunedH.append(conseq)
    return prunedH  # 返回一个满足最小可信度要求的关联规则表


# 合并上一次生成的关联规则，生成新的候选关联规则列表
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # 合并关联规则
        Hmp1 = aprioriGen(H, m + 1)  # 创建新的频繁项集
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # 新的关联规则表中，每条关联规则中的右边Hmp1必须包含两个元素
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


# 打印关联规则
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("可信度: %f" % ruleTup[2])


if __name__ == "__main__":
    dataSet = loadDataSet()
    C1 = createC1(dataSet)  # 原始集合C1
    print("原始集合C1:", C1)
    D = list(map(set, dataSet))
    print("数据集D:", D)
    L1, supportData0 = scanD(D, C1, minSupport=0.5)
    print("频繁项集L1:", L1, "\n支持度:", supportData0)
    print("---------------完整测试-----------------")
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet)
    print("频繁项集L:", L, "\n支持度: ", supportData)
    print('--------------关联规则的生成-------------')
    rules = generateRules(L, supportData, minConf=0.5)
    print("关联规则为:", rules)
