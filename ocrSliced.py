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
import random


def p(x): print(x)


def sparse2dense(x): return x[1]
slicedImgWidth=30

characterListFull = string.ascii_lowercase + string.ascii_uppercase + " .,\n" + string.digits + string.punctuation  # "'!?\"-:()"
characterListInUsage = characterListFull
# A|MOVE|to|stop|Mr.|Gaitskell|from
inputstring = "A MOVE to stop Mr .Gaitskell from."
inputimageName = "ocrdata/a01-000u-s00-00.png"
valiDir = "validata/"

num_features = 11  # bigger -> worse!
num_classes = len(characterListInUsage) + 1 + 1  # Accounting the 0th indice +  space + blank label = 28 characters


def img2tensor(imageNdarr_imread, labelStr):
    imgRaw_ = imageNdarr_imread.transpose()
    print("raw image")
    print(imgRaw_.shape)
    rawW_ = imgRaw_.shape[0]
    rawH_ = imgRaw_.shape[1]
    imgHeight_ = num_features
    imgWidth_ = int(round(rawW_ * (imgHeight_ / rawH_)))
    imgResized = misc.imresize(imgRaw_, (imgWidth_, imgHeight_))

    transposedImgNdarr = np.asarray(imgResized)
    normalizedNdarr = (transposedImgNdarr - np.mean(transposedImgNdarr)) / np.std(transposedImgNdarr)
    p(labelStr)
    p('normalizedImgNdarr')
    p(normalizedNdarr.shape) # 1 168 11
    p("sliced")
    sliced=sliceImg(normalizedNdarr,slicedImgWidth)
    p(sliced.shape) # 46,30 11
    return sliced, [sliced.shape[1]], sparse_tuple_from([np.asarray([characterListInUsage.index(x) for x in labelStr])])


def sliceImg(imgInParams, cnnWidth):
    # img shape is ,wid,hei   (142, 11)
    # (1,w,h) -> x=[(cnnw,h)] -> (1,len x,x)
    # need (1,len,(..))
    step=3
    imglen = len(imgInParams) # w
    slicedNdarr=[]
    iters=range(0, imglen - cnnWidth, step)
    p("sliceImg:"+str(iters)+","+str(imgInParams.shape)+",imglen:"+str(imglen))
    for r in iters:
        # p("sliceImg")
        # p(str(r))
        item=imgInParams[r:cnnWidth+r,:]
        # p(item.shape)
        slicedNdarr.append(item)
    slicedNdarr=np.array(slicedNdarr)
    p('sliced item:'+str(slicedNdarr.shape))
    res=[x.flatten() for x in slicedNdarr]
    p(np.array([res]).shape)
    return np.array([res])

def cnn(sink_x):
    # input_layer = tf.reshape(sink_x, [-1, 30, 11, 1])
    conv1 = tf.layers.conv2d(
        inputs=sink_x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    output = tf.reshape(pool2, [-1, 7 * 7 * 64])

    return output

def biLstmCnnCtcGraph():
    num_hidden = 100
    initial_learning_rate = 1e-3

    graph = tf.Graph()
    with graph.as_default():
        sink_x = tf.placeholder(tf.float32, [None, 30, num_features])  # num feature is input length?
        # afterCnn=[cnn(xx) for xx in sink_x]
        def process(x):
            p('processinga')
            return cnn(x)

        # afterCnn=tf.map_fn(process,sink_x)
        # afterCnn=[process(i) for i in sink_x]
        afterCnn=tf.map_fn(lambda x:process(x),sink_x)
        afterCnn=np.array(afterCnn)

        p('after cnn ok,shape:'+str(afterCnn.shape))
        sink_lenth_x = tf.placeholder(tf.int32, [None])
        sink_y = tf.sparse_placeholder(tf.int32)  # targets
        #
        # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # stack = tf.contrib.rnn.MultiRNNCell(
        #     [tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)] ,
        #     state_is_tuple=True)
        # frnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # brnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # frnn = tf.contrib.rnn.GRUCell(num_hidden)
        # brnn = tf.contrib.rnn.GRUCell(num_hidden)]

        frnn=tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.GRUCell(num_hidden) for i in [1, 1]])
        brnn = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.GRUCell(num_hidden) for i in [1, 1]])

        # outputs, _ = tf.nn.dynamic_rnn(stack, sink_x, sink_lenth_x, dtype=tf.float32)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(frnn, brnn, afterCnn, sink_lenth_x, dtype=tf.float32)
        sink_x_shape = tf.shape(sink_x)
        batch_s, max_timesteps = sink_x_shape[0], sink_x_shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden])
        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))

        cost = tf.reduce_mean(tf.nn.ctc_loss(sink_y, logits, sink_lenth_x))
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sink_lenth_x)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, sink_lenth_x)

        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sink_y))
        return sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph


def train(datalist, valilist):
    num_epochs = 1500
    batch_size = 1
    num_examples = 1
    sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph = biLstmCnnCtcGraph()

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        vald = valilist[0]
        # vald=datalist[0]
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            ler_accum = 0
            ler_avg = 1
            start = time.time()
            for idx, datalistRandom in enumerate(datalist):
                feed = {sink_x: datalist[0],
                        sink_lenth_x: datalist[1],
                        sink_y: datalist[2]
                        }

                batch_cost, _ = sess.run([cost, optimizer], feed)
                # train_cost += batch_cost * batch_size
                # train_ler += sess.run(ler, feed_dict=feed) * batch_size
                train_cost = batch_cost * batch_size
                train_ler = sess.run(ler, feed_dict=feed) * batch_size
                ler_accum += train_ler
                ler_avg = ler_accum / len(datalistRandom)

                train_cost /= num_examples
                train_ler /= num_examples

                val_cost, val_ler = sess.run([cost, ler], feed_dict=feed)
                print(str(curr_epoch) + "," + str(idx))
                print("Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, accumLer = {:.3f},time = {:.3f}"
                      .format(curr_epoch + 1, num_epochs, train_cost, train_ler, ler_avg,
                              val_cost, val_ler, time.time() - start)
                      )
                feed2 = {sink_x: vald[0],
                         sink_lenth_x: vald[1],
                         sink_y: vald[2]
                         }
                result_sparse = sess.run(decoded[0], feed_dict=feed2)
                costvalid, lerValid = sess.run([cost, ler], feed_dict=feed2)
                print('Original:\n%s' % ''.join([characterListInUsage[i] for i in sparse2dense(vald[2])]))
                print('Decoded:\n%s' % ''.join([characterListInUsage[i] for i in sparse2dense(result_sparse)]))
                p('ler : %s' % lerValid)
            if ler_accum < 0.01:
                break


import os

labelFileName = "sentences.txt"


def labelFile2list(imageFN):
    crimefile = open(labelFileName, 'r')
    return [line.replace('\n', '').split(' ') for line in crimefile.readlines()]


def imageFilename2label(list, imageFileName_):
    return [x for x in list if x[0] == imageFileName_][0]


def dir2finalDataList(imgDir):
    imgFilename_labelList = []
    for x in list(map(lambda adir: adir.split('.')[0], os.listdir(imgDir))):
        foundLabel = imageFilename2label(labelFile2list(labelFileName), x)
        if foundLabel[2] == 'ok':
            imgFilename_labelList.append([x + ".png", foundLabel[9].replace('|', ' ')])
    p(imgFilename_labelList)

    finalfeedable = [[x[0], x[1], x[2]] for x in
                     [img2tensor(misc.imread(imgDir + y[0]), y[1]) for y in imgFilename_labelList]]

    return finalfeedable


def mainSingle():
    inputstring = "A MOVE to stop Mr .Gaitskell from."
    inputimageName = "ocrdata/a01-000u-s00-00.png"
    x1, y1, y1len = img2tensor(misc.imread(inputimageName), inputstring)
    train([[x1, y1, y1len]])


def mainf():
    datalist = dir2finalDataList("ocrdata/")
    # valilist = dir2finalDataList("validata/")
    train(datalist, datalist)

mainf()
# dir2finalDataList("validata/")