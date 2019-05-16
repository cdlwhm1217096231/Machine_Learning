#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-16 14:52:27
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from numpy import *

# CRAT回归
"""
    普通的决策树算法：首先选择出最佳的特征进行划分，然后根据特征的取值将数据分成几份，该特征使用后再后面的过程中将不再使用。
    将数据集切分成多分容易建模的数据,如果首次切分后仍难以拟合线性模型就继续切分
    切分使用的是二元切分的方法，二元切分即每次把数据集切成两份，如果数据的某特征值等于要切分所要求的值，那么这些数据就进入树的左子树，反之进入树的右子树;使用二元切分的方法来处理连续型变量。
    回归树：每个叶子节点包含单个值
    模型树：其中每个节点包含一个线性方程
"""
# 树回归


def loadDataSet(fileName):
    """
        将每行映射成浮点数
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = map(float, curLine)
        dataMat.append(list(fltLine))
    return dataMat


# 数据集进行二分
def binSplitDataSet(dataset, feature, value):
    """
        feature:待切分的特征
        value:该特征的某个取值
    """
    mat0 = dataset[nonzero(dataset[:, feature] > value), :][0]
    mat1 = dataset[nonzero(dataset[:, feature] <= value), :][0]
    return mat0, mat1


# CART回归树生成叶子节点函数
def regLeaf(dataset):
    return mean(dataset[:, -1])


# 回归树的误差计算
# 最好特征选择依据：样本y的均方差MSE*样本总数 = 总方差，作为当前样本情况下的混乱程度
def regErr(dataset):
    return var(dataset[:, -1]) * shape(dataset)[0]


# 模型树的误差计算
def modelErr(dataset):
    ws, X, Y = linearSolve(dataset)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


# 线性回归模型
def linearSolve(dataset):
    m, n = shape(dataset)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataset[:, 0:n - 1]
    Y = dataset[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0:
        raise NameError("矩阵是奇异矩阵，不可逆")
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 模型树的叶子节点生成函数
def modelLeaf(dataset):
    ws, X, Y = linearSolve(dataset)
    return ws


# 核心代码：选择最佳特征和阈值
def chooseBestSplit(dataset, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
        leafType:建立叶子节点的函数
        errType:误差计算函数
        tolS,tolN:用来控制函数什么时候停止划分
        用最佳方式切分数据集和生成相应的叶子节点
    """
    tolS = ops[0]   # 容许的误差下降值---防止过拟合，预剪枝
    tolN = ops[1]   # 切分的最小样本数---防止过拟合，预剪枝
    # 统计不同剩余特征值的数量，如果特征只剩1个，则不需要再划分了，直接返回---情况1：停止划分条件
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataset)
    m, n = shape(dataset)
    S = errType(dataset)  # 计算当前划分下的混乱程度，随着不断的划分，混乱程度应该是下降
    bestS = inf  # 最小误差
    bestIndex = 0  # 最佳特征
    bestValue = 0  # 最佳划分阈值
    for featureIndex in range(n - 1):  # 总共有n-1个特征
        for splitVal in set(dataset[:, featureIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(
                dataset, featureIndex, splitVal)  # 基于此阈值进行样本二分
            # 不适合的划分方法
            if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)  # 当前的误差
            if newS < bestS:
                bestIndex = featureIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:  # 情况2：停止划分的条件----划分后，混乱程度几乎没有下降，则不划分了
        return None, leafType(dataset)
    mat0, mat1 = binSplitDataSet(dataset, bestIndex, bestValue)  # 正式开始划分
    # 情况3：停止划分的条件--对切分后得到的两个子集进mat0,mat1进行检查，如果某个子集的大小小于用户自定义的参数tolN时就停止划分
    if ((shape(mat0)[0] < tolN)) or (shape(mat1)[0] < tolN):
        return None, leafType(dataset)
    # 返回特最佳特征的索引和对应的最佳划分阈值
    return bestIndex, bestValue


# 树构建的函数
def createTree(dataset, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
        leafType:建立叶子节点的函数
        errType:误差计算函数
        ops:包含树构建过程中所需其他参数的元组
    """
    feature, val = chooseBestSplit(dataset, leafType, errType, ops)
    # 树停止构建的条件
    if feature == None:
        return val
    retTree = {}
    retTree['spInd'] = feature  # 最佳特征对应的索引
    retTree['spVal'] = val  # 最佳特征索引列对应的特征的划分阈值
    lSet, rSet = binSplitDataSet(dataset, feature, val)  # 二元划分数据集
    # 开始递归，创建左右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 测试输入变量是否是一棵树(判断当前的节点是否是叶子节点)，返回是bool类型
def isTree(obj):
    return (type(obj).__name__ == "dict")


# 从上向下，递归查找两个叶子节点
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2  # 找到两个叶子节点后，计算它们的平均值


"""
    一棵树的节点过多时，表示该模型可能对数据进行了过拟合。使用测试集上的交叉验证技术来发现是否已经过拟合。
    剪枝：通过降低决策树的复杂度来避免过拟合的过程
    预剪枝：在函数chooseBestSplit()中的提前终止条件，实际上是进行一种所谓的预剪枝
    后剪枝：使用测试集进行剪枝操作
    后剪枝的具体步骤：首先，指定参数，使得构建出的树足够大，足够复杂，便于剪枝；然后，从上而下找到叶子节点，用测试集来判断将这些叶子节点合并是否能降低测试误差，是的话就进行合并操作。
"""


# 后剪枝操作------代剪枝的树，剪枝的测试数据来自测试集
def prune(tree, testData):
    if shape(testData)[0] == 0:  # 确认测试数据是否是空的？
        return getMean(tree)
    # 检测某个分支是叶子节点还是子树？
    if (isTree(tree["right"]) or isTree(tree["left"])):  # 是子树，
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if (isTree(tree['left'])):  # 是子树，就调用函数来对该子树进行剪枝
        tree['left'] = prune(tree['left'], lSet)
    if (isTree(tree['right'])):
        tree['right'] = prune(tree['right'], rSet)
    # 经过N此迭代，找到了叶子节点，开始合并
    # 还需检查对左右两个分支完成剪枝后，还需要检查它们是否仍然是子树？两个分支不是子树时，就进行合并操作
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 没有合并前的误差和
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
            sum(power(rSet[:, -1] - tree['right'], 2))
        # 合并后的误差和
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# 回归树评估函数
def regTreeEval(model, inDat):
    return float(model)


# 模型树评估函数
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


# 测试
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 测试数据的输入接口函数
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == "__main__":
    print("------------------test1----------------")
    myDat = loadDataSet("E:\\Code\\local_code\\机器学习实战\\DataSet\\ex00.txt")
    myDat = mat(myDat)
    retTree = createTree(myDat)
    print("构建好的回归树1:", retTree)
    print("------------------test2----------------")
    myDat = loadDataSet("E:\\Code\\local_code\\机器学习实战\\DataSet\\ex0.txt")
    myDat = mat(myDat)
    retTree = createTree(myDat)
    print("构建好的回归树1:", retTree)
    # 构建树的算法createTree()函数其实对输入参数tolS、tolN非常敏感，使用其他值可能不太容易达到好的效果
    print("------------------test3----------------")
    myDat1 = loadDataSet("E:\\Code\\local_code\\机器学习实战\\DataSet\\ex2.txt")
    myDat1 = mat(myDat1)
    tree = createTree(myDat1, ops=(0, 1))
    print("构建好的回归树2:", tree)
    myDatTest = loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\ex2test.txt")
    myDatTest = mat(myDatTest)
    prune_tree = prune(tree, myDatTest)
    print("后剪枝后的回归树:", prune_tree)
    print("----------------test4-------------")
    myDat2 = loadDataSet("E:\\Code\\local_code\\机器学习实战\\DataSet\\exp2.txt")
    myDat2 = mat(myDat2)
    model_tree = createTree(myDat2, modelLeaf, modelErr, ops=(1, 10))
    print("模型树:", model_tree)
    print("----------------test5模型比较-------------")
    trainMat = mat(loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\bikeSpeedVsIq_train.txt"))
    testMat = mat(loadDataSet(
        "E:\\Code\\local_code\\机器学习实战\\DataSet\\bikeSpeedVsIq_test.txt"))
    # 创建回归树模型
    myTree = createTree(trainMat, ops=(1, 20))
    y_hat = createForeCast(myTree, testMat[:, 0])  # 预测
    print("回归树模型相关系数", corrcoef(y_hat, testMat[:, 1], rowvar=0)[0, 1])
    # 创建模型树
    myTree = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    y_hat = createForeCast(myTree, testMat[:, 0], modelTreeEval)  # 预测
    print("模型树模型相关系数", corrcoef(y_hat, testMat[:, 1], rowvar=0)[0, 1])
    # 创建线性回归模型
    ws, X, Y = linearSolve(trainMat)
    for i in range(shape(trainMat)[0]):
        y_hat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print("线性回归模型相关系数", corrcoef(y_hat, testMat[:, 1], rowvar=0)[0, 1])
