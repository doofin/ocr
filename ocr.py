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
def sparse2dense(x):return x[1]

characterListFull = string.ascii_lowercase + string.ascii_uppercase + " .,\n"+string.digits+string.punctuation
characterListInUsage = characterListFull
# A|MOVE|to|stop|Mr.|Gaitskell|from
inputstring = "A MOVE to stop Mr .Gaitskell from."
inputimageName = "ocrdata/a01-000u-s00-00.png"
valiDir = "validata/"

num_features = 10  # bigger -> worse!
num_classes = len(characterListInUsage) + 1 + 1  # Accounting the 0th indice +  space + blank label = 28 characters


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
    p(labelStr)
    label_dense = np.asarray([characterListInUsage.index(x) for x in labelStr])
    return normalizedImgNdarr, sparse_tuple_from([label_dense]), [normalizedImgNdarr.shape[1]]


def train(datalist,valilist):
    num_epochs = 200
    num_hidden = 50
    num_layers = 1
    batch_size = 1
    initial_learning_rate = 1e-2
    momentum = 0.9
    num_examples = 1
    num_batches_per_epoch = int(num_examples / batch_size)

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
            for idx, dataset in enumerate(datalist):
                feed = {sink_y: dataset[0],
                        sink_x: dataset[1],
                        sink_lenth_y: dataset[2]}

                batch_cost, _ = sess.run([cost, optimizer], feed)
                train_cost += batch_cost * batch_size
                train_ler += sess.run(ler, feed_dict=feed) * batch_size
                train_cost /= num_examples
                train_ler /= num_examples
                val_cost, val_ler = sess.run([cost, ler], feed_dict=feed)
                print(str(curr_epoch) + "," + str(idx))
                log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
                                 val_cost, val_ler, time.time() - start))

        vald=valilist[0]
        feed2 = {sink_y: vald[0],
                sink_x: vald[1],
                sink_lenth_y: vald[2]}
        result_sparse = sess.run(decoded[0], feed_dict=feed2)
        # result_dense = result_sparse[1]
        # final_decoded = [characterListInUsage[i] for i in sparse2dense(result_sparse)]
        print('Original:\n%s' % ''.join([characterListInUsage[i] for i in sparse2dense(vald[1])]))
        print('Decoded:\n%s' % ''.join([characterListInUsage[i] for i in sparse2dense(result_sparse)]))


# p(list(map(lambda x: x, range(1, 10))))  # wtf
# x1, y1, y1len = img2tensor(misc.imread(inputimageName), inputstring)
# train([[x1, y1, y1len],[x1, y1, y1len]])

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
        foundLabel = imageFilename2label(labelFile2list(labelFileName), x)[9]
        imgFilename_labelList.append([x + ".png", foundLabel.replace('|', ' ')])
    p(imgFilename_labelList)

    finalfeedable = [[x[0], x[1], x[2]] for x in
                     [img2tensor(misc.imread(imgDir + y[0]), y[1]) for y in imgFilename_labelList]]
    return finalfeedable

def mainSingle():
    inputstring = "A MOVE to stop Mr .Gaitskell from."
    inputimageName = "ocrdata/a01-000u-s00-00.png"
    x1, y1, y1len = img2tensor(misc.imread(inputimageName), inputstring)
    train([[x1,y1,y1len]])
def mainf():
    # imgFilename_labelList=[]
    # for x in list(map(lambda adir:adir.split('.')[0],os.listdir("ocrdata/"))):
    #     foundLabel = imageFilename2label(labelFile2list(labelFileName), x)[9]
    #     imgFilename_labelList.append([x+".png",foundLabel.replace('|',' ')])
    # p(imgFilename_labelList)
    #
    # # feedable=[img2tensor(misc.imread("ocrdata/"+x[0]),x[1]) for x in imgFilename_labelList]
    # finalfeedable=[ [x[0],x[1],x[2]] for x in [img2tensor(misc.imread("ocrdata/"+x[0]),x[1]) for x in imgFilename_labelList]]
    datalist=dir2finalDataList("ocrdata/")
    valilist = dir2finalDataList("validata/")
    train(datalist,valilist)


mainf()
# mainSingle()


# foundLabel=imageFilename2label(labelFile2list(labelFileName), "n04-139-s01-01")[9]
# p(foundLabel)
# # ok or not : 2 , sentense : 9
# for imageFilename in os.listdir("ocrdata/"):
#     print(imageFilename)
