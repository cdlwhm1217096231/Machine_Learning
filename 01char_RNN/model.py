# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


class CharRNN:
    def __init__(self, num_classes, num_seqs=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()


# 定义输入数据
    def build_inputs(self):
        '''
        self.inputs是外部传入的一个batch内的输入数据,形状是(self.num_seqs, self.num_steps)
        self.num_seqs是一个batch内句子的个数，相当于batch_size
        self.num_steps是每个句子的长度
        self.targets是self.inputs对应的训练目标，形状和self.inputs相同，内容是self.inputs每个字母对应的下一个字母
        self.keep_prob是dropout所需要的概率，训练时keep_prob=0.5;测试时keep_prob=1
        '''
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 对于中文，需要使用embedding层
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:   # self.use_embedding参数控制是否使用embedding
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):   # 指定模型运行的具体设备，cpu不区分设备号，统一使用/cpu:0;GPU使用/gpu:0和/gpu:1区分
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])  # 定义一个embedding变量，此变量也是可以训练的，通过训练得到embedding的具体数值
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)  # self.lstm_inputs是直接输入lstm的数据,embedding_size是embedding空间的大小


    # 构建多层lstm模型
    def build_lstm(self):
        # 创建单个cell
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)    # lstm_size是lstm隐藏层的大小
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop
        # 单个cell堆叠多层
        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)])  #  构建多层RNN，num_layers是LSTM的层数
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')


    # 定义损失
    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)


    # 优化
    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    # 定义训练过程
    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss, self.final_state, self.optimizer], feed_dict=feed)
                end = time.time()
                # control the print lines
                if step % log_every_n == 0:   # log_every_n是运行参数，每隔10次会在屏幕上打出日志
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):  # save_every_n也是运行参数，每隔1000步会将模型保存下来
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)


    # 测试时,选择一个字符当作起始字符，使用训练好的模型得到对应的下一个字符的概率。根据这个概率，选择一个字符输出，并将该字符作为下一个序列的输入，再生成下一个字符。依次重复上述步骤
    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,self.keep_prob: 1., self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
