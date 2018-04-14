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
import datetime

isValidating = True


def p(x): print(x)


def joinStr(strList): return ''.join(strList)


def replaceCharlist(toReplaceCharList, str): return ''.join([' ' if c in toReplaceCharList else c for c in str])


def sparse2dense(x): return x[1]


starttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
slicedImgWidth = 30
characterBasic = string.ascii_lowercase + " " + string.digits
characterExtra = ".,\n|" + string.punctuation
characterListInUsage = characterBasic
characterListForDecoding = characterListInUsage + characterExtra
num_features = 10  # bigger -> worse!
num_classes = len(characterListInUsage) + 1 + 1  # Accounting the 0th indice +  space + blank label = 28 characters

valiDir = "validata/"


def img2tensor(imageNdarr_imread, labelStr, fn):
    p("img2tensor:" + labelStr + ",," + fn)
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


def biLstmCtcGraph(is_validating):
    num_hidden = 200
    initial_learning_rate = 1e-4

    graph = tf.Graph()
    with graph.as_default():
        sink_x = tf.placeholder(tf.float32, [None, None, num_features])  # num feature is input length?
        sink_lenth_x = tf.placeholder(tf.int32, [None])
        sink_y = tf.sparse_placeholder(tf.int32)  # targets

        # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # stack = tf.contrib.rnn.MultiRNNCell([
        #     tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)] ,state_is_tuple=True)

        # stack = tf.contrib.rnn.MultiRNNCell([
        #     tf.contrib.rnn.GRUCell(num_hidden)
        #     for _ in [1, 1, 1, 1, 1]])
        # tf.nn.rnn_cell.GRUCell(num_hidden)
        stackTrain = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.LSTMCell(num_units=prob_numHidden[1], use_peepholes=True),
                                          output_keep_prob=prob_numHidden[0])
            for prob_numHidden in [[0.5, 400], [0.6, 300], [0.8, 200], [0.8, 200]]
        ])
        stackValid = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.LSTMCell(num_units=prob_numHidden[1], use_peepholes=True)
            for prob_numHidden in [[0.5, 400], [0.6, 300], [0.8, 200], [0.8, 200]]
        ])
        stack = stackValid if is_validating else stackTrain

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
        source_y_decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, sink_lenth_x)

        ler = tf.reduce_mean(tf.edit_distance(tf.cast(source_y_decoded[0], tf.int32), sink_y))
        saver = tf.train.Saver()
        return sink_x, sink_lenth_x, sink_y, source_y_decoded, cost, optimizer, ler, graph, saver


def train(datalist, valilist):
    num_epochs = 500
    batch_size = 1
    num_examples = 1
    sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph, saver = biLstmCtcGraph()
    minimalLer = 1
    # saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        # writer = tf.summary.FileWriter("/tmp/tflog", sess.graph)
        tf.global_variables_initializer().run()
        # vald = valilist[0]
        # vald=datalist[0]
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            ler_accum = 0
            ler_avg = 1
            start = time.time()
            datalistRandom = datalist
            random.shuffle(datalistRandom)
            lenofdatalist = len(datalist)
            for idx, datalistRandom in enumerate(datalistRandom):
                feed = {sink_x: datalistRandom[0],
                        sink_lenth_x: datalistRandom[1],
                        sink_y: datalistRandom[2]}

                batch_cost, _ = sess.run([cost, optimizer], feed)
                train_cost = batch_cost * batch_size
                train_ler = sess.run(ler, feed_dict=feed) * batch_size
                ler_accum += train_ler
                ler_avg = ler_accum / len(datalistRandom)

                train_cost /= num_examples
                train_ler /= num_examples

                # val_cost, val_ler = sess.run([cost, ler], feed_dict=feed)
                validAvgLer = 0
                if (idx % 5 == 0):
                    print(str(idx) + '/' + str(lenofdatalist) + " of data")
                    print("Epoch {}/{},train_cost {:.3f}, train_ler {:.3f}, accumLer {:.3f},time = {:.3f}"
                          .format(curr_epoch + 1, num_epochs, train_cost, train_ler, ler_avg, time.time() - start))

                    p(starttime + ' ----> ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    valilistInuse = valilist[:3]
                    for aValid in valilistInuse:
                        feed2 = {sink_x: aValid[0],
                                 sink_lenth_x: aValid[1],
                                 sink_y: aValid[2]
                                 }
                        result_sparse = sess.run(decoded[0], feed_dict=feed2)
                        _, lerValid = sess.run([cost, ler], feed_dict=feed2)

                        if (lerValid < minimalLer):
                            minimalLer = lerValid
                            if (lerValid < 0.7):
                                p("saving model....")
                                saver.save(sess, "saved/model-" + str(lerValid) + ".ckpt")
                        print('Original:\n%s' % joinStr([characterListInUsage[i] for i in sparse2dense(aValid[2])]))
                        print('Decoded:\n%s' % joinStr([characterListInUsage[i]
                                                        if (i < len(characterListInUsage))
                                                        else characterListInUsage[len(characterListInUsage) - 1]
                                                        for i in sparse2dense(result_sparse)]))
                        p('ler : ' + str(lerValid) + ',minimal:' + str(minimalLer))
                        validAvgLer += lerValid
                        avgValidLer = validAvgLer / len(valilistInuse)
                    p('------avg ler:' + str(avgValidLer) + '-------')
                    validAvgLer = 0
                    p("\n")
    # writer.close()


def validate(valilist):
    sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph, saver = biLstmCtcGraph(True)
    with tf.Session(graph=graph) as sess:
        # tf.global_variables_initializer().run()
        saver.restore(sess, "saved/model-0.18421052.ckpt")
        minimalLer = 1
        for aValid in valilist:
            result_sparse, lerValid = sess.run([decoded[0], ler], feed_dict={sink_x: aValid[0],
                                                                             sink_lenth_x: aValid[1],
                                                                             sink_y: aValid[2]})
            print('Original:\n%s' % joinStr([characterListInUsage[i] for i in sparse2dense(aValid[2])]))
            print('Decoded:\n%s' % joinStr([characterListInUsage[i]
                                            if (i < len(characterListInUsage))
                                            else characterListInUsage[len(characterListInUsage) - 1]
                                            for i in sparse2dense(result_sparse)]))
            p('ler : ' + str(lerValid) + ',minimal:' + str(minimalLer))
            validAvgLer = 0
            validAvgLer += lerValid
            avgValidLer = validAvgLer / len(valilist)
        p('------avg ler:' + str(avgValidLer) + '-------')


import os

labelFileName = "sentences.txt"


def labelFile2list(imageFN):
    labFile = open(labelFileName, 'r')
    return [line.replace('\n', '').split(' ') for line in labFile.readlines()]


def imageFilename2label(list, imageFileName_):
    return [x for x in list if x[0] == imageFileName_][0]


def dir2finalDataList(imgDir):
    p("reading...")
    imgNameAndLabel = []  # [(imgname,label)]
    ct = 0
    labelfilelist = labelFile2list(labelFileName)
    for x in list(map(lambda adir: adir.split('.')[0], os.listdir(imgDir))):
        foundLabel_line = imageFilename2label(labelfilelist, x)
        if foundLabel_line[2] == 'ok':
            a_label = foundLabel_line[9]
            cleaned_label = replaceCharlist(characterExtra, a_label).lower()
            p('reading nth:' + str(ct) + ',,label is : ' + str(cleaned_label))
            imgNameAndLabel.append([x + ".png", cleaned_label])
            ct += 1

    # p(imgNameAndLabel)
    p("read complete")
    # medianBlur
    finalfeedable = [[x[0], x[1], x[2]] for x in
                     [img2tensor(
                         cv2.medianBlur(cv2.threshold(cv2.imread(imgDir + y[0], 0), 210, 255, cv2.THRESH_BINARY)[1], 5),
                         y[1],
                         y[0]
                     ) for y in imgNameAndLabel]]

    return finalfeedable


def mainSingle():
    inputstring = "A MOVE to stop Mr .Gaitskell from."
    inputimageName = "ocrdata/a01-000u-s00-00.png"
    x1, y1, y1len = img2tensor(misc.imread(inputimageName), inputstring)
    train([[x1, y1, y1len]])


traindata = "ocrdata/"


def mainf():
    datalist = dir2finalDataList(traindata)
    valilist = dir2finalDataList("validata/")
    train(datalist, valilist)


def mainValid():
    validate(dir2finalDataList("validata/"))


mainValid()
# mainf()
# dir2finalDataList("validata/")
