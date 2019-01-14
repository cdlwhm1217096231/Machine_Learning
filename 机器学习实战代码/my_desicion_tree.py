# -*- coding: UTF-8 -*-
from math import log
import operator

# 原始训练数据集D


def createDataSet():
    dataSet = dataSet = dataSet = [[0, 0, 0, 0, 'no'],   # 训练数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet,labels

# 将数据集D根据某一特征划分为不同的数据子集


def splitDataSet(dataSet,axis,value):
    retDataSet  = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducefeatVec = featVec[0:axis]
            reducefeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducefeatVec)
    return retDataSet

# 计算信息熵的函数


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本总数
    labelsCounts = {}          # 该空字典用来记录每个类别出现的次数
    for featVec in dataSet:
        currentLabel = featVec[-1]   # dataSet中的每个元素均是列表featVec, featVec列表的最后一个元素是分类的类别标签
        if currentLabel not in labelsCounts.keys():   # 如果数据集D中的分类标签currentLabel未出现在labelsCounts字典的key中，则原始labelsCounts的value值为0
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1       # 如果数据集D中的分类标签currentLabel已经出现在labelsCounts字典的key中，则原始labelsCounts的value值加1
    shannonEnt = 0.0   # 初始化信息熵为0
    for value in labelsCounts.values():  # 遍历labelsCounts字典中的存放每个类别出现的次数
        p = float(value) / numEntries
        shannonEnt -= p * log(p, 2)  # 计算D的信息熵
    return shannonEnt

# 选择最优特征


def selectBestFeature(dataSet):
    numFeatures = len(dataSet[1]) - 1
    baseShannonEnt = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # i = 0时，则是选取第一行的第一列元素，也就是第一个特征的所有取值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            p = len(subDataSet) / float(len(dataSet))
            newEntropy += p*calcShannonEnt(subDataSet)  # 计算条件熵
        infoGain = baseShannonEnt - newEntropy
        print('第%d个特征的信息增益是: %.3f' % (i, infoGain))
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature, bestInfoGain

# 多数表决法决定无法确定的叶子节点类别,结束递归


def majorityClass(classList):
    classLabelsCount = {}
    for keys in classList.keys():
        if keys not in classLabelsCount.keys():
            classLabelsCount[keys] = 0
            classLabelsCount[keys] += 1
    sortedClassLablesCount = sorted(classLabelsCount.values(), key=operator.itemgetter(0), reverse=True)  # 根据classLabelsCount.items()对象的第1个域进行降序排列（即value值进行降序排列）
    return sortedClassLablesCount[0]
# sorted()函数说明:
# sorted(iterable[, cmp[, key[, reverse]]]) 其中key的参数为一个函数或者lambda函数。所以itemgetter可以用来当key的参数
# operator.itemgetter(1) operator.itemgetter()函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值

# 创建决策树的程序


def createTree(dataSet, labels, featLabels):  # labels包含所有特征的名字，具体见训练数据集D的格式, featLabels用来存放最优特征名字的列表
    classList = [example[-1] for example in dataSet]  # classList这个列表存放了所有的样本的类别  即是否发放贷款
    if classList.count(classList[0]) == len(classList):  # 递归结束的第一种条件： 所有叶子节点的类别都相同，递归程序结束
        return classList[0]
    if len(dataSet[0]) == 1:   # 递归结束的第二种条件: 使用完所有特征后，仍然无法将数据集划分为某一确定的类别，则采用多数表决法返回出现次数最多的类标签，确定叶子节点的类别
        return majorityClass(classList)
    bestFeatureIndex, bestInfoGain = selectBestFeature(dataSet)  # 最优特征的索引号bestFeatureIndex，最大信息增益bestInfoGain
    bestFeatureName = labels[bestFeatureIndex]   # 获取最优特征的名字
    featLabels.append(bestFeatureName)
    """下面是产生的决策树字典表现形式程序段"""
    myTree = {bestFeatureName: {}}  # 根据每层的最优特征名字生成决策树
    del(labels[bestFeatureIndex])           # 删除已经使用过的特征
    beatFeatValues = [example[bestFeatureIndex] for example in dataSet]  # 得到训练集D中所有最优特征的特征取值
    uniqueVals = set(beatFeatValues)
    for value in uniqueVals:
        subLables = labels[:]   # 所有特征名字的复制,但是labels列表已经更新
        myTree[bestFeatureName][value] = createTree(splitDataSet(dataSet, bestFeatureIndex, value), subLables, featLabels)
    return myTree

# 测试算法函数


def classify(inputTree, featLabels, testVec):      # inputTree是已经训练得到的决策树、featLabels是最优特征名字组成的列表、testVec是测试向量
    firstStr = next(iter(inputTree))   # next()是获得下一条数据的函数 iter()迭代器函数：获取可迭代对象即已经生成的决策树myTree  firstStr：第一个最优特征的名字
    secondDict = inputTree[firstStr]   # 第一个最优特征的名字作为第二个字典开始的key值，然后得到value值即开始进入第二个字典
    featIndex = featLabels.index(firstStr)  # 第一个最优特征的索引
    for key in secondDict.keys():           # 第二个字典的key值即第二个最优特征名字
        if testVec[featIndex] == key:       # 比较testVec变量中的值与树节点的值
            if type(secondDict[key]).__name__ == 'dict':  # 未到达叶子节点
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]    # 到达叶子节点，返回当前节点的分类标签
    return classLabel



if __name__ == '__main__':
    myData, labels = createDataSet()
    print(myData)
    print('选取第二个特征axis=1划分数据集，特征的取值是value=0:\n', splitDataSet(myData, 1, 0))
    print('训练集D的信息熵为:', calcShannonEnt(myData))
    print('总结：第%d个特征是最优特征且信息增益是：%.3f' % (selectBestFeature(myData)))
    featLabels = []                          # 存放所有最优特征名字的列表
    myTree = createTree(myData, labels, featLabels)
    print('决策树的建立:', myTree)
    testVec = [0, 1]                         # 测试数据只选用前两个特征，也可以使用四个特征
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('测试结果:放贷')
    if result == 'no':
        print('测试结果:不放贷')

