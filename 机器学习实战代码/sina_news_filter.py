# -*- coding: utf-8 -*-
"""
GaussianNB     先验为高斯分布的朴素贝叶斯，
MultinomialNB  先验为多项式分布的朴素贝叶斯，
BernoulliNB    先验为伯努利分布的朴素贝叶斯

使用sklearn库中的朴素贝叶斯分类算法，分类新浪新闻
对于新闻分类，属于多分类问题，我们可以使用MultinamialNB()完成我们的新闻分类问题
class sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
alpha=1.0，拉普拉斯平滑系数

fit_prior   class_prior              最终先验概率

False       填或不填没有意义           P(Y = Ck) = 1 / k
True        不填                      P(Y = Ck) = mk / m
True        填                        P(Y = Ck) = class_prior

MultinomialNB一个重要的功能是有partial_fit方法，这个方法的一般用在如果训练集数据量非常大，一次不能全部载入内存的时候。
这时我们可以把训练集分成若干等分，重复调用partial_fit来一步步的学习训练集，非常方便

 在使用MultinomialNB的fit方法或者partial_fit方法拟合数据后，我们可以进行预测。
 预测有三种方法，包括predict，predict_log_proba和predict_proba。
 predict方法就是我们最常用的预测方法，直接给出测试集的预测类别输出。
 predict_proba则不同，它会给出测试集样本在各个类别上预测的概率。predict_proba预测出的各个类别概率里的最大值对应的类别，也就是predict方法得到类别。
 predict_log_proba它会给出测试集样本在各个类别上预测的概率的一个对数转化，转化后predict_log_proba预测出的各个类别对数概率里的最大值对应的类别，也就是predict方法得到类别。

"""
# -*- coding: UTF-8 -*-
import os
import jieba
import random
import operator
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# 中文文本处理
"""
将所有文本分成训练集和测试集，并对训练集中的所有单词进行词频统计，并按降序排序
"""


def chineseParse(folder_path, text_size=0.2):  # test_size:测试集占比，默认占所有数据集的百分之20
    # 查看folder_path下的文件
    folder_list = os.listdir(folder_path)
    data_list = []                                                # 训练集
    class_list = []

    # 遍历每个子文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(
            folder_path, folder)        # 根据子文件夹，生成新的路径
        # 存放子文件夹下的txt文件的列表
        files = os.listdir(new_folder_path)

        j = 1
        # 遍历每个txt文件
        for file in files:
            if j > 100:                                            # 每类txt样本数最多100个
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:    # 打开txt文件
                raw = f.read()

            word_cut = jieba.cut(raw, cut_all=False)  # 精简模式，返回一个可迭代的generator
            word_list = list(word_cut)  # generator转换为list

            data_list.append(word_list)    # 添加数据集数据
            class_list.append(folder)      # 添加数据集类别
            j += 1
        print('data_list:\n', data_list)
        print('class_list:\n', class_list)
    # zip()将数据与标签对应形成一组元组，放入列表中,成为列表中的一个元素
    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)   # 将data_class_list随机排序
    index = int(len(data_class_list) * text_size) + 1  # 按照索引值划分训练集和测试集
    train_list = data_class_list[index:]  # 训练集
    test_list = data_class_list[:index]   # 测试集
# *解压zip(),将每个元组中的第一个位置的元素取出来，每个元组的第二个位置的元素取出来，形成两个新的元组，放入列表中，成为列表中的两个新元素
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)
    all_words_dict = {}  # 统计训练集中每个单词出现的频率
    for word_data_list in train_data_list:
        for word in word_data_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1


# 根据键的值降序排列
    all_words_tuple_list = sorted(all_words_dict.items(), key=operator.itemgetter(1), reverse=True)
    all_words_list, all_words_num = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, train_class_list, test_data_list, test_class_list


# 为了消除高频的逗号、&等符号和‘的、了、在’、数字对分类结果的影响,所以制定规则，来消除上述几种字符对新闻分类的影响
# 规则如下:
"""
1. 首先去掉高频词，至于去掉多少个高频词，我们可以通过观察去掉高频词个数和最终检测准确率的关系来确定
2. 去除数字，不把数字作为分类特征
3. 去除一些特定的词语，比如：”的”，”了”，”在”，”不”，”当然”,”怎么”这类的对新闻分类无影响的介词、代词、连词
"""
# 利用已经整理好的stopwords_cn.txt文本，来除去频率出现太高的连词、数字、标点符号等
# 读取文件里的内容，并去重


def makeWordSet(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()   # 逐行读出文件
        for line in lines:
            word = line.strip()  # 除去文件末尾的回车符
            if len(word) > 0:
                words_set.add(word)
    return words_set


# 文本特征选取
"""
all_words_list - 训练集所有文本列表
deleteN - 删除单词出现频率最高的deleteN个词
stopwords_set - 指定的结束语
feature_words - 已经过滤掉对分类不起作用的单词，得到的特征集
"""


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度为1000
            break
# 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

# 根据feature_words将文本向量化


def textFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        # 出现在特征集中，则置1
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list                # 返回结果


# 新闻分类器
"""
train_feature_list - 训练集向量化的特征文本
test_feature_list - 测试集向量化的特征文本
train_class_list - 训练集分类标签
test_class_list - 测试集分类标签
test_accuracy - 分类器精度
"""


def textClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

# 计算分类器精度


def ave(c):
    return sum(c) / len(c)


if __name__ == '__main__':
    folder_path = 'E:/机器学习/ml_ws/SogouC/Sample'
    all_words_list, train_data_list, train_class_list, test_data_list, test_class_list = chineseParse(folder_path, text_size=0.2)
    # all_words_list就是将所有训练集的切分结果通过词频降序排列构成的单词合集
    print('all_words_list:\n', all_words_list)
    # 生成stopwords_set
    stopwords_file = 'E:/机器学习/ml_ws/stopwords_cn.txt'
    stopwords_set = makeWordSet(stopwords_file)
    feature_words = words_dict(all_words_list, 100, stopwords_set)
    print('feature_words:\n', feature_words)

    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)  # 0 20 40 60 ... 980
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = textFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = textClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)
# 绘制出了deleteNs和test_accuracy的关系，这样我们就可以大致确定去掉前多少的高频词汇
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()
# 通过deleteNs与test_accuracy之间的关系图，确定删除450个单词时，分类器效果最好
    test_accuracy_list = []
    feature_words = words_dict(all_words_list, 450, stopwords_set)
    train_feature_list, test_feature_list = textFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = textClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    print('分类器精度:', ave(test_accuracy_list))
