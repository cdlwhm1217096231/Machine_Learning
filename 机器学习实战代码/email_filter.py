# -*- coding: UTF-8 -*-
import numpy as np
import random
import re

'''
朴素贝叶斯过滤垃圾邮件
两个文件夹ham和spam，spam文件下的txt文件为垃圾邮件
'''
# 对于英文文本，我们可以以非字母、非数字作为符号进行切分，使用split函数即可
# 将字符串转化为字符列表


def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)                          # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]       # 除了单个字母，例如大写的I，其它单词变成小写
# 创建词汇表


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
# 根据词汇表，将输入向量inputSet进行数字化，用0，1表示输入向量中的单词是否在词汇表中出现过


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)									# 创建一个其中所含元素都为0的向量
    for word in inputSet:												# 遍历每个词条
        if word in vocabList:											# 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else: print("单词:%s不在词汇表中!" % word)
    return returnVec

# 根据词汇表，构建词袋模型


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)										# 创建一个其中所含元素都为0的向量
    for word in inputSet:												# 遍历每个词条
        if word in vocabList:											# 如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec													# 返回词袋模型
# 朴素贝叶斯分类器训练函数


def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)							# 计算训练的文档数目
    numWords = len(trainMatrix[0])							# 计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)		# 文档属于侮辱类的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)	                            # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0                        	                # 分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:							# 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:												# 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)							# 取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive							# 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
# 朴素贝叶斯分类器分类函数


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    	# 对应元素相乘,logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
# 测试朴素贝叶斯分类器


def testNB():
    docList = []
    classList = []
    for i in range(1, 26):                                                              # 遍历25个txt文件
        wordList = textParse(open('E:/机器学习/ml_ws/spam/%d.txt' % i, 'r').read())     # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)                                                 # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('E:/机器学习/ml_ws/spam/%d.txt' % i, 'r').read())      # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)                                                 # 标记非垃圾邮件，1表示垃圾文件
    print('没有除去重复单词的词汇表:\n', docList)
    vocabList = createVocabList(docList)                                    # 创建词汇表，不重复
    print('已经除去重复单词的词汇表:\n', vocabList)
    print('已除去重复单词的词汇表中含有的单词个数:', len(vocabList))
    # 留存交叉验证：随机选择一部分数据作为测试集，而剩余部分作为训练集
    trainingSet = list(range(50))
    testingSet = []                                                          # 创建存储训练集索引值的列表和测试集索引值的列表
    for i in range(10):                                                     # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))      # uniform()方法将随机生成下一个实数，它在 [x, y) 范围内.随机选取测试集的索引值
        testingSet.append(trainingSet[randIndex])                              # 添加测试集的索引值
        del(trainingSet[randIndex])                                         # 在训练集列表中删除已经添加到测试集的索引值
    trainMat = []                                            # 创建训练集矩阵
    trainClasses = []                                        # 创建训练集类别标签向量
    for docIndex in trainingSet:                                            # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))       # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])                            # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0                                                          # 错误分类计数
    for docIndex in testingSet:                                                # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])           # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:    # 如果分类错误
            errorCount += 1                                                 # 错误计数加1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testingSet) * 100))


if __name__ == '__main__':
    testNB()