import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import collections

# '''
# python 3.6.4
# tensorflow 1.5.0
# '''

sess = tf.Session()

# 分词后的文本路径
save_data = 'fenci.conv'

# 声明算法模型参数
batch_size = 500
embedding_size = 200
vocabulary_size = 5000
generations = 50000
module_learning = 0.001
window_size = 3

save_embeddings_every = 5000
print_loss_every = 100


def load_data():
    data = []
    with open(os.path.join(save_data), 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            data.append(line)
    f.close()
    return data


texts = load_data()
texts = [x for x in texts if len(x.split(' ')) > 2 * window_size + 1]  # 设置句子长度大于3个单词

# 创建词典
word_dict = {}


# 创建单词字典：一个 词->索引 字典和一个 索引->词 字典
# 想要验证单词集中的每个单词时，可以用逆序单词字典
def build_dictionary():
    # 从text列表中取出每一行，再从每一行中取出每一个词
    words = [word for line in texts for word in line.strip().split(' ')]
    # print(words)

    # 初始化[词，词频]列表，每个词开始的词频为unknown
    count = [['RARE', -1]]
    # 把取出的词加入count，如果该词未加入，则添加，最多添加vocabulary_size的单词
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    for word, word_count in count:
        word_dict[word] = len(word_dict)


build_dictionary()
# 构建逆序词典
word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))


# '''
# print(word_dict)
# print(word_dict_rev)
# 词典中的单词如下，从单词词频大到小排序
# {'RARE': 0, '你': 1, '我': 2, '，': 3, '的': 4, '。': 5, '了': 6, '是': 7, '！': 8, ',': 9, '=': 10, '不': 11, '啊': 12,...}
# {0: 'RARE', 1: '你', 2: '我', 3: '，', 4: '的', 5: '。', 6: '了', 7: '是', 8: '！', 9: ',', 10: '=', 11: '不', 12: '啊',...}
# '''

# 把文本中的词比对词典，找到该词索引
def text2number():
    index_data = []
    for sentence in texts:
        sentence_data = []
        # 取出每个词，对于该词存在于词典，则取出索引，否则为0，为未知词
        for word in sentence.strip().split(' '):
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        index_data.append(sentence_data)
    return index_data


text_data = text2number()

# 初始化待拟合单词嵌套并声明算法模型的数据占位符　
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
x_input = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])

# 创建循环将窗口内所有单词嵌套
embed = tf.zeros([batch_size, embedding_size])
for element in range(2 * window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_input[:, element])

# 使用nce损失函数
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed,
                                     labels=y_target, num_sampled=int(batch_size / 2), num_classes=vocabulary_size))


def generate_batch_data():
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # 选择随机的句子开始
        rand_sentence_ix = int(np.random.choice(len(text_data), size=1))
        rand_sentence = text_data[rand_sentence_ix]
        # 取出句子后，移动窗口去查看嵌套词
        window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, x in
                            enumerate(rand_sentence)]
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

        batch_and_label = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]
        batch_and_label = [(x, y) for x, y in batch_and_label if len(x) == 2 * window_size]
        batch, label = [list(x) for x in zip(*batch_and_label)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(label[:batch_size])
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    return batch_data, label_data


saver = tf.train.Saver({'embeddings': embeddings})

# 定义优化器
opt = tf.train.GradientDescentOptimizer(module_learning).minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

# 遍历迭代训练
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_label = generate_batch_data()
    feed_dict = {x_input: batch_inputs, y_target: batch_label}
    sess.run(opt, feed_dict=feed_dict)
    #  返回loss值
    if (i + 1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        print('Loss at step {} : {}'.format(i + 1, loss_val))
    if (i + 1) % save_embeddings_every == 0:
        with open(os.path.join('fenci_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dict, f)

        model_checkpoint_path = os.path.join(os.getcwd(), 'cbow_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))
