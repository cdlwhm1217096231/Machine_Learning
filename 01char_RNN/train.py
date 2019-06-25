import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch')  # 每个batch中含有的句子数量
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')   # 每个序列的长度
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')  # lstm层含有的隐藏层大小
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')   # 2层lsm堆叠
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')  # 是否使用词嵌入
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')  # 词嵌入空间的大小
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')  # 训练时候的dropout参数的设置
tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')    # 最大训练次数
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')  # 运行参数，每1000次保存一次模型
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps') # 运行参数，每10次输出一次结果到屏幕
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')    # 词典的最大容量


def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    print(converter.vocab_size)
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
