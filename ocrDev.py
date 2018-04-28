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
import cv2
import datetime
import matplotlib.pyplot as plt
import os


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
num_width = 24  # bigger -> worse! 50 feature?
num_firstLayer = 400
num_classes = len(characterListInUsage) + 1 + 1  # Accounting the 0th indice +  space + blank label = 28 characters

valiDir = "validata/"
isValidating = True
modelLabel = "devC" + str(num_width)
savedir = "saved/gru4l" + str(num_width) + "-" + str(num_firstLayer) + "/"
statsdir = "stats/"
shape1, shape1name = [[0.6, num_firstLayer], [0.7, 300], [0.8, 200], [0.9, 200]], "sdfsdfsdf"
shape2 = [[0.6, num_firstLayer], [0.7, 300], [0.7, 300], [0.8, 200], [0.8, 200], [0.9, 200]]


def img2tensor(imgreaded, labelStr, fn):
    p("img2tensor:" + labelStr + ",," + fn)

    imageNdarr_imread = cv2.medianBlur(cv2.threshold(imgreaded, 210, 255, cv2.THRESH_BINARY)[1], 5)
    # imageNdarr_imread = cv2.threshold(imgreaded, 210, 255, cv2.THRESH_BINARY)[1]
    # imageNdarr_imread=imgreaded
    imgRaw_ = imageNdarr_imread.transpose()
    # print("raw image")
    # print(imgRaw_.shape)
    rawW_ = imgRaw_.shape[0]
    rawH_ = imgRaw_.shape[1]
    imgHeight_ = num_width
    imgWidth_ = int(round(rawW_ * (imgHeight_ / rawH_)))
    imgResized = misc.imresize(imgRaw_, (imgWidth_, imgHeight_))
    # imgResized = cv2.threshold(imgResized, 210, 255, cv2.THRESH_BINARY)[1]
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


def nncell(hu): return tf.nn.rnn_cell.GRUCell(num_units=hu)


def biLstmCtcGraph(is_validating):
    num_hidden = 200
    initial_learning_rate = 1e-4

    graph = tf.Graph()
    with graph.as_default():
        sink_x = tf.placeholder(tf.float32, [None, None, num_width])  # num feature is input length?
        sink_lenth_x = tf.placeholder(tf.int32, [None])
        sink_y = tf.sparse_placeholder(tf.int32)  # targets
        stackTrain = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.DropoutWrapper(cell=nncell(prob_numHidden[1]),
                                          output_keep_prob=prob_numHidden[0])
            for prob_numHidden in shape1
        ])
        stackValid = tf.nn.rnn_cell.MultiRNNCell([
            nncell(prob_numHidden[1])
            for prob_numHidden in shape1
        ])
        stack = stackValid if is_validating else stackTrain
        outputs, _ = tf.nn.dynamic_rnn(stack, sink_x, sink_lenth_x, dtype=tf.float32)
        sink_x_shape = tf.shape(sink_x)
        batch_s = sink_x_shape[0]
        outputs_reshaped = tf.reshape(outputs, [-1, num_hidden])
        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes]))  # stddev=0.2
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        logits = tf.transpose(tf.reshape(tf.matmul(outputs_reshaped, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))
        cost = tf.reduce_mean(tf.nn.ctc_loss(sink_y, logits, sink_lenth_x))
        optimizer = tf.train.AdamOptimizer(initial_learning_rate, 0.9).minimize(cost)
        source_y_decoded, _ = tf.nn.ctc_beam_search_decoder(logits, sink_lenth_x)
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(source_y_decoded[0], tf.int32), sink_y))
        saver = tf.train.Saver()
        return sink_x, sink_lenth_x, sink_y, source_y_decoded, cost, optimizer, ler, graph, saver


def current_milli_time(): return int(round(time.time() * 1000))


def train(datalist, valilist):
    num_epochs = 500
    sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph, saver = biLstmCtcGraph(is_validating=False)
    minimalLer = 1
    totalsteps = 0
    os.makedirs(statsdir, exist_ok=True)
    validfile = open(statsdir + "valid.txt", "w")
    trainfile = open(statsdir + "train.txt", "w")

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for curr_epoch in range(num_epochs):
            ler_accum = 0
            startTime = time.time()
            lenOfDatalist = len(datalist)
            for idx, a_data in enumerate(datalist):
                feed = {sink_x: a_data[0], sink_lenth_x: a_data[1], sink_y: a_data[2]}
                train_cost, _, train_ler = sess.run([cost, optimizer, ler], feed)
                ler_accum += train_ler
                ler_avg = ler_accum / len(a_data)
                p("nth total : ---------: " + str(totalsteps))
                trainfile.write(str(totalsteps) + ',' + str(current_milli_time()) + ',' + str(train_ler) + '\n')
                trainfile.flush()
                totalsteps += 1
                validAccumLer = 0
                if (idx % 5 == 0):
                    print(str(idx) + '/' + str(lenOfDatalist) + " of data")
                    print("Epoch {}/{},train_cost {:.3f}, train_ler {:.3f}, accumLer {:.3f},time = {:.3f}"
                          .format(curr_epoch + 1, num_epochs, train_cost, train_ler, ler_avg, time.time() - startTime))

                    p(starttime + ' ----> ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    valilistInuse = valilist[:5]
                    validAvgLer = 0
                    for aValid in valilistInuse:
                        feed2 = {sink_x: aValid[0], sink_lenth_x: aValid[1], sink_y: aValid[2]}
                        lerValid, result_sparse = sess.run([ler, decoded[0]], feed_dict=feed2)
                        print('Original:\n%s' % joinStr([characterListInUsage[i] for i in sparse2dense(aValid[2])]))
                        print('Decoded:\n%s' % joinStr([characterListInUsage[i]
                                                        if (i < len(characterListInUsage))
                                                        else characterListInUsage[len(characterListInUsage) - 1]
                                                        for i in sparse2dense(result_sparse)]))
                        p('ler : ' + str(lerValid) + ',minimal:' + str(minimalLer))
                        validAccumLer += lerValid
                        validAvgLer = validAccumLer / len(valilistInuse)
                    validfile.write(str(totalsteps) + ',' + str(current_milli_time()) + ',' + str(validAvgLer) + '\n')
                    validfile.flush()
                    p('------avg ler:' + str(validAvgLer) + '-------')
                    if (validAvgLer < minimalLer):
                        minimalLer = validAvgLer
                        if (validAvgLer < 0.6):
                            p("saving model....,avg ler:" + str(validAvgLer) + ",, minimal : " + str(minimalLer))
                            os.makedirs(savedir, exist_ok=True)
                            saver.save(sess, savedir + str(validAvgLer) + ".ckpt")
                    p("\n")


def getLabelList():
    labFile = open("sentences.txt", 'r')
    return [line.replace('\n', '').split(' ') for line in labFile.readlines()]


def imageFilename2label(list, imageFileName_):
    return [x for x in list if x[0] == imageFileName_][0]


def dir2finalDataList(imgDir):
    p("reading...")
    imgNameAndLabel = []  # [(imgname,label)]
    ct = 0
    labelfilelist = getLabelList()
    for x in list(map(lambda adir: adir.split('.')[0], os.listdir(imgDir))):
        foundLabel_line = imageFilename2label(labelfilelist, x)
        if foundLabel_line[2] == 'ok':
            a_label = foundLabel_line[9]
            cleaned_label = replaceCharlist(characterExtra, a_label).lower()
            p('reading nth:' + str(ct) + ',,label is : ' + str(cleaned_label))
            imgNameAndLabel.append([x + ".png", cleaned_label])
            ct += 1

    p("read complete")
    finalfeedable = [[x[0], x[1], x[2]] for x in
                     [img2tensor(
                         cv2.imread(imgDir + y[0], 0),
                         y[1],
                         y[0]
                     ) for y in imgNameAndLabel]]

    return finalfeedable


def validate(valilist, savedmodel):
    sink_x, sink_lenth_x, sink_y, decoded, cost, optimizer, ler, graph, saver = biLstmCtcGraph(True)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, savedmodel)
        minimalLer = 1
        validAvgLerAccum = 0
        nth = 0
        lenv = len(valilist)
        for aValid in valilist:
            result_sparse, lerValid = \
                sess.run([decoded[0], ler], feed_dict={sink_x: aValid[0], sink_lenth_x: aValid[1], sink_y: aValid[2]})
            p(str(nth) + " th" + "total: " + str(lenv))
            print('Original:\n%s' % joinStr([characterListInUsage[i] for i in sparse2dense(aValid[2])]))
            print('Decoded:\n%s' % joinStr([characterListInUsage[i]
                                            if (i < len(characterListInUsage))
                                            else characterListInUsage[len(characterListInUsage) - 1]
                                            for i in sparse2dense(result_sparse)]))
            nth += 1
            validAvgLerAccum += lerValid
            avgValidLer = validAvgLerAccum / nth

            p('ler : ' + str(lerValid) + ',minimal:' + str(minimalLer))
            p('------avg ler:' + str(avgValidLer) + '-------')
        p('------avg ler final:' + str(avgValidLer) + '-------')
        return avgValidLer


def mainValid():
    avgErrlist = []
    for m in ["0.31876641511917114.ckpt"]:
        p("===============> using model:::::" + m + " <---------------------------------------")
        r = validate(dir2finalDataList("validata/"), "saved/gru34-400/" + m)
        avgErrlist.append(str(r))

    map(p, avgErrlist)


def mainf():
    datalist = dir2finalDataList("ocrdata/")
    valilist = dir2finalDataList("validata/")
    train(datalist, valilist)


mainf()
# mainValid()
# dir2finalDataList("validata/")

# frnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
# brnn=tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

# frnn = tf.contrib.rnn.GRUCell(num_hidden)
# brnn = tf.contrib.rnn.GRUCell(num_hidden)
# outputs, _ = tf.nn.bidirectional_dynamic_rnn(frnn, brnn, sink_x, sink_lenth_x, dtype=tf.float32)
