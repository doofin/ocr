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

    transposedImgNdarr = np.asarray(imgResized[np.newaxis, :])
    normalizedImgNdarr = (transposedImgNdarr - np.mean(transposedImgNdarr)) / np.std(transposedImgNdarr)
    p(labelStr)
    label_dense = np.asarray([characterListInUsage.index(x) for x in labelStr])
    p('normalizedImgNdarr')
    p(normalizedImgNdarr.shape) # 1 168 11
    p(normalizedImgNdarr[0].shape)
    p(normalizedImgNdarr[0][0].shape)
    p("sliced")
    sliced=sliceImg(normalizedImgNdarr,slicedImgWidth)
    p(sliced.shape) # 46,30 11
    return sliced, [sliced.shape[1]], sparse_tuple_from([label_dense])


def sliceImg(imgInParams, cnnWidth):
    # img shape is ,wid,hei   (142, 11)
    # (1,w,h) -> x=[(cnnw,h)] -> (1,len x,x)
    # need (1,len,(..))
    step=3
    img=imgInParams[0] # w h
    imglen = len(img) # w
    sliceList=[]
    iters=range(0, imglen - cnnWidth, step)
    p("sliceImg:"+str(iters)+","+str(img.shape)+",imglen:"+str(imglen))
    for r in iters:
        # p("sliceImg")
        # p(str(r))
        item=img[r:cnnWidth+r,:]
        # p(item.shape)
        sliceList.append(item)
    sliceList=np.array(sliceList)
    res=[x.flatten() for x in sliceList]
    return np.array([res])
def cnn():
    sink_x = tf.placeholder(tf.float32, [slicedImgWidth, num_features])  #
    input_layer = tf.reshape(sink_x, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
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
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    return sink_x,pool2_flat

def lstmCnnGraph():
    num_hidden = 82
    num_layers = 1
    initial_learning_rate = 1e-2

    graph = tf.Graph()
    with graph.as_default():
        sink_x = tf.placeholder(tf.float32, [None, None, num_features])  #
        sink_lenth_x = tf.placeholder(tf.int32, [None])
        sink_y = tf.sparse_placeholder(tf.int32)  # targets

        input_layer = tf.reshape(sink_x, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
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
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(stack, sink_x, sink_lenth_x, dtype=tf.float32)

        sink_x_shape = tf.shape(sink_x)
        batch_s, max_timesteps = sink_x_shape[0], sink_x_shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden])
        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))

        cost = tf.reduce_mean(tf.nn.ctc_loss(sink_y, logits, sink_lenth_x))
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sink_lenth_x)

        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sink_y))
        return sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph
def biLstmCtcSlicedGraph():
    num_hidden = 130
    initial_learning_rate = 1e-2
    # (1,w,h) -> x=[(cnnw,h)] -> (1,len x,x)
    graph = tf.Graph()
    with graph.as_default():
        sink_x = tf.placeholder(tf.float32, [1, None, slicedImgWidth,num_features])  # num feature is input length?
        sink_lenth_x = tf.placeholder(tf.int32, [None])
        sink_y = tf.sparse_placeholder(tf.int32)  # targets

        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)] ,
            state_is_tuple=True)
        frnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        brnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # outputs, _ = tf.nn.dynamic_rnn(stack, sink_x, sink_lenth_x, dtype=tf.float32)
        outputs,_=tf.nn.bidirectional_dynamic_rnn(frnn,brnn,sink_x,sink_lenth_x,dtype=tf.float32)
        sink_x_shape = tf.shape(sink_x)
        batch_s, max_timesteps = sink_x_shape[0], sink_x_shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden])
        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))

        cost = tf.reduce_mean(tf.nn.ctc_loss(sink_y, logits, sink_lenth_x))
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sink_lenth_x)

        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sink_y))
        return sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph

def biLstmCtcGraph():
    num_hidden = 130
    initial_learning_rate = 1e-2

    graph = tf.Graph()
    with graph.as_default():
        sink_x = tf.placeholder(tf.float32, [None, None, num_features])  # num feature is input length?
        sink_lenth_x = tf.placeholder(tf.int32, [None])
        sink_y = tf.sparse_placeholder(tf.int32)  # targets

        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)] ,
            state_is_tuple=True)
        frnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        brnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # outputs, _ = tf.nn.dynamic_rnn(stack, sink_x, sink_lenth_x, dtype=tf.float32)
        outputs,_=tf.nn.bidirectional_dynamic_rnn(frnn,brnn,sink_x,sink_lenth_x,dtype=tf.float32)
        sink_x_shape = tf.shape(sink_x)
        batch_s, max_timesteps = sink_x_shape[0], sink_x_shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden])
        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))

        cost = tf.reduce_mean(tf.nn.ctc_loss(sink_y, logits, sink_lenth_x))
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sink_lenth_x)

        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sink_y))
        return sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph


def train(datalist, valilist):
    num_epochs = 1500
    batch_size = 1
    num_examples = 1
    sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph = biLstmCtcSlicedGraph()

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
    # datalist = dir2finalDataList("ocrdata/")
    valilist = dir2finalDataList("validata/")
    train(valilist, valilist)

mainf()
# dir2finalDataList("validata/")