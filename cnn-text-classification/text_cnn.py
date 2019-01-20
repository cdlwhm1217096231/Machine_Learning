import tensorflow as tf
import numpy as np


"""TextCNN模型"""


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
            sequence_length:输入句子的长度，经过padding后，句子的统一长度是59
            num_classes:最终的分类的类别总数
            vocab_size:词汇表的总次数
            embedding_size:词嵌入的维度
            filter_sizes:卷积核的覆盖的单词数量,例如[3, 4, 5]意味着我们分别用3,4,5个单词组成一个卷积核，故总的卷积核格式3*num_filters
            num_filters:每种尺寸卷积核的个数
            l2_reg_lambda:不使用L2正则化，正则化参数初始化为0
        """

        # 定义输入、输出、dropout占位符变量
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x")  # None表示每次训练或测试时输入的数据大小是任意的
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(   # 训练时，dropout的概率大小，防止过拟合
            tf.float32, name="dropout_keep_prob")

        # 保持跟踪L2正则化的loss的值(可选)
        l2_loss = tf.constant(0.0)

        # 词嵌入层
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                # 使用服从正态分布随机初始化权重变量
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # 创建词嵌入的操作，得到3D的tensor(None, sequence_length, embedding_size)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)   # tf中的conv2d卷积操作函数接收4D的tensor(batch, width, height, channel),由于词嵌入层的输出结果不包含channel这个维度，所以手动增加一个维度

        # 为每一个卷积核创建一个卷积和池化层，使用的卷积核尺寸size是不同的
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",  # narrow convolution，不进行填充0操作
                    name="conv")
                # 执行卷积操作后，输出的矩阵尺寸(1, sequence_length-filter_size+1, 1, 1)
                # 激活函数进行激活
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 进行最大池化操作
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # 执行max-pooling操作后，每个特定尺寸的卷积核上的输出尺寸为(batch_size, 1, 1, num_filters)
                pooled_outputs.append(pooled)

        # 将每个卷积核经过max-pooling之后的结果拼接起来,得到的shape是(batch_size, num_filters_total)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)  # 拼接操作
        self.h_pool_flat = tf.reshape(
            self.h_pool, [-1, num_filters_total])  # 送入softmax之前，进行flatten

        # 训练时,增加一个dropout操作，防止过拟合，此操作在softmax层之前;测试时，此参数设置为1.0,不进行dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())  # 初始化全连接层的权重
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.h_drop, W, b, name="scores")  # 此函数执行wx+b操作
            self.predictions = tf.argmax(
                self.scores, 1, name="predictions")  # 选择输出概率最大的类别作为预测类别

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
