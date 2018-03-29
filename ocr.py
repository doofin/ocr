from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import numpy as np
from six.moves import xrange as range
from utils import sparse_tuple_from as sparse_tuple_from
from scipy import misc
import string


def p(x): print(x)


characterListFull = string.ascii_lowercase + string.ascii_uppercase + " .,\n"
characterListInUsage = characterListFull
p(characterListInUsage)
# A|MOVE|to|stop|Mr.|Gaitskell|from
inputstring = "A MOVE to stop Mr .Gaitskell from."
inputimageName = "ocrdata/a01-000u-s00-00.png"

num_features = 13  # bigger -> worse!
num_classes = len(characterListInUsage) + 1 + 1  # Accounting the 0th indice +  space + blank label = 28 characters

num_epochs = 10
num_hidden = 50
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-2
momentum = 0.9

num_examples = 1
num_batches_per_epoch = int(num_examples / batch_size)


def img2tensor(imageNdarr_imread, labelStr):
    imgRaw_ = imageNdarr_imread.transpose()
    print("raw image")
    print(imgRaw_.shape)
    rawW_ = imgRaw_.shape[0]
    rawH_ = imgRaw_.shape[1]
    imgHeight_ = num_features
    imgWidth_ = int(round(rawW_ * (imgHeight_ / rawH_)))
    imgTensor_ = misc.imresize(imgRaw_, (imgWidth_, imgHeight_))

    imgNdarr = np.asarray(imgTensor_[np.newaxis, :])
    transposedImgNdarr = imgNdarr
    normalizedImgNdarr = (transposedImgNdarr - np.mean(transposedImgNdarr)) / np.std(transposedImgNdarr)

    label_dense = np.asarray([characterListInUsage.index(x) for x in labelStr])
    return normalizedImgNdarr, sparse_tuple_from([label_dense]), [normalizedImgNdarr.shape[1]]


def train(datalist):
    graph = tf.Graph()
    with graph.as_default():
        sink_x = tf.sparse_placeholder(tf.int32)
        sink_y = tf.placeholder(tf.float32, [None, None, num_features])  # num feature is input length?
        sink_lenth_y = tf.placeholder(tf.int32, [None])
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(stack, sink_y, sink_lenth_y, dtype=tf.float32)
        shape = tf.shape(sink_y)
        batch_s, max_timesteps = shape[0], shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden])
        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))
        loss = tf.nn.ctc_loss(sink_x, logits, sink_lenth_y)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sink_lenth_y)
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sink_x))
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()
            for idx,dataset in enumerate(datalist):
                p("nth dataset")
                print(idx)
                feed = {sink_y: dataset[0],
                        sink_x: dataset[1],
                        sink_lenth_y: dataset[2]}

                batch_cost, _ = sess.run([cost, optimizer], feed)
                train_cost += batch_cost * batch_size
                train_ler += sess.run(ler, feed_dict=feed) * batch_size
                train_cost /= num_examples
                train_ler /= num_examples
                val_cost, val_ler = sess.run([cost, ler], feed_dict=feed)
                log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
                                 val_cost, val_ler, time.time() - start))
                idx+=1

        result_dec = sess.run(decoded[0], feed_dict=feed)
        result_dense = result_dec[1]
        final_decoded = [characterListInUsage[i] for i in result_dense]
        print('Original:\n%s' % inputstring)
        print('Decoded:\n%s' % ''.join(final_decoded))


p(list(map(lambda x: x, range(1, 10))))  # wtf
x1, y1, y1len = img2tensor(misc.imread(inputimageName), inputstring)
train([[x1, y1, y1len],[x1, y1, y1len]])