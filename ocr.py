from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
from six.moves import xrange as range
from utils import sparse_tuple_from as sparse_tuple_from
from scipy import misc
import string
import random
import cv2


def p(x): print(x)


def joinStr(strList): return ''.join(strList)


def replaceCharlist(toReplaceCharList, str): return ''.join([' ' if c in toReplaceCharList else c for c in str])


def sparse2dense(x): return x[1]


slicedImgWidth = 30
characterBasic = string.ascii_lowercase + " " + string.digits
characterExtra = ".,\n|" + string.punctuation
characterListInUsage = characterBasic
characterListForDecoding = characterListInUsage + characterExtra
num_features = 12  # bigger -> worse!
num_classes = len(characterListInUsage) + 1 + 1  # Accounting the 0th indice +  space + blank label = 28 characters

valiDir = "validata/"


def img2tensor(imageNdarr_imread, labelStr):
    imgRaw_ = imageNdarr_imread.transpose()
    # print("raw image")
    # print(imgRaw_.shape)
    rawW_ = imgRaw_.shape[0]
    rawH_ = imgRaw_.shape[1]
    imgHeight_ = num_features
    imgWidth_ = int(round(rawW_ * (imgHeight_ / rawH_)))
    imgResized = misc.imresize(imgRaw_, (imgWidth_, imgHeight_))
    imgResized = cv2.threshold(imgResized, 210, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(imgResized)
    # plt.show()

    transposedImgNdarr = np.asarray(imgResized[np.newaxis, :])
    normalizedImgNdarr = (transposedImgNdarr - np.mean(transposedImgNdarr)) / np.std(transposedImgNdarr)
    # p(labelStr)
    label_dense = np.asarray([characterListInUsage.index(x) for x in labelStr])
    # p('normalizedImgNdarr')
    # p(normalizedImgNdarr.shape)

    # len(normalizedImgNdarr[0][0]) == num features =11
    # len(normalizedImgNdarr[0]) == img width
    # p(len(normalizedImgNdarr[0]))
    # sliceImg(normalizedImgNdarr)
    return normalizedImgNdarr, [normalizedImgNdarr.shape[1]], sparse_tuple_from([label_dense])


def biLstmCtcGraph():
    num_hidden = 180
    initial_learning_rate = 1e-4

    graph = tf.Graph()
    with graph.as_default():
        sink_x = tf.placeholder(tf.float32, [None, None, num_features])  # num feature is input length?
        sink_lenth_x = tf.placeholder(tf.int32, [None])
        sink_y = tf.sparse_placeholder(tf.int32)  # targets
        #
        # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # stack = tf.contrib.rnn.MultiRNNCell([
        #     tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)] ,state_is_tuple=True)

        stack = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.GRUCell(num_hidden)
            for _ in [1, 1, 1, 1]])
        # stack=tf.contrib.rnn.GRUCell(num_hidden)

        outputs, _ = tf.nn.dynamic_rnn(stack, sink_x, sink_lenth_x, dtype=tf.float32)

        # frnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # brnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

        # frnn = tf.contrib.rnn.GRUCell(num_hidden)
        # brnn = tf.contrib.rnn.GRUCell(num_hidden)
        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(frnn, brnn, sink_x, sink_lenth_x, dtype=tf.float32)

        sink_x_shape = tf.shape(sink_x)
        batch_s, max_timesteps = sink_x_shape[0], sink_x_shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden])

        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.2))
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))

        cost = tf.reduce_mean(tf.nn.ctc_loss(sink_y, logits, sink_lenth_x))
        # optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
        optimizer = tf.train.AdamOptimizer(initial_learning_rate, 0.9).minimize(cost)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sink_lenth_x)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, sink_lenth_x)

        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sink_y))
        return sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph


def train(datalist, valilist):
    num_epochs = 500
    batch_size = 1
    num_examples = 1
    sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph = biLstmCtcGraph()

    with tf.Session(graph=graph) as sess:
        # writer = tf.summary.FileWriter("/tmp/tflog", sess.graph)
        tf.global_variables_initializer().run()
        vald = valilist[0]
        # vald=datalist[0]
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            ler_accum = 0
            ler_avg = 1
            start = time.time()
            datalistRandom = datalist
            random.shuffle(datalistRandom)
            for idx, datalistRandom in enumerate(datalistRandom):
                feed = {sink_x: datalistRandom[0],
                        sink_lenth_x: datalistRandom[1],
                        sink_y: datalistRandom[2]
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
                print('Decoded:\n%s' % ''.join([characterListInUsage[i]
                                                if (i<len(characterListInUsage))
                                                else characterListInUsage[len(characterListInUsage)-1]
                                                for i in sparse2dense(result_sparse)]))
                p('ler : %s' % lerValid)
                p("'\n\n'")
            if ler_accum < 0.01:
                break

    # writer.close()


import os

labelFileName = "sentences.txt"


def labelFile2list(imageFN):
    labFile = open(labelFileName, 'r')
    return [line.replace('\n', '').split(' ') for line in labFile.readlines()]


def imageFilename2label(list, imageFileName_):
    return [x for x in list if x[0] == imageFileName_][0]


def dir2finalDataList(imgDir):
    p("reading...")
    imgNameAndLabel = []
    ct = 0
    for x in list(map(lambda adir: adir.split('.')[0], os.listdir(imgDir))):
        foundLabel_line = imageFilename2label(labelFile2list(labelFileName), x)
        if foundLabel_line[2] == 'ok':
            a_label = foundLabel_line[9]
            cleaned_label = replaceCharlist(characterExtra, a_label).lower()
            p('reading nth:' + str(ct) + ',,label is : ' + str(cleaned_label))
            imgNameAndLabel.append([x + ".png", cleaned_label])
            ct += 1

    p(imgNameAndLabel)
    p("read complete")
    # medianBlur
    finalfeedable = [[x[0], x[1], x[2]] for x in
                     [img2tensor(
                         cv2.medianBlur(cv2.threshold(cv2.imread(imgDir + y[0], 0), 210, 255, cv2.THRESH_BINARY)[1], 5),
                         y[1]) for y in imgNameAndLabel]]

    return finalfeedable


def mainSingle():
    inputstring = "A MOVE to stop Mr .Gaitskell from."
    inputimageName = "ocrdata/a01-000u-s00-00.png"
    x1, y1, y1len = img2tensor(misc.imread(inputimageName), inputstring)
    train([[x1, y1, y1len]])


def mainf():
    datalist = dir2finalDataList("ocrdata/")
    valilist = dir2finalDataList("validata/")
    train(datalist, valilist)


mainf()
# dir2finalDataList("validata/")
