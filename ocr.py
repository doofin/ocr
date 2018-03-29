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


num_features = 13  # bigger -> worse!
num_classes = len(characterListInUsage) + 1 + 1  # Accounting the 0th indice +  space + blank label = 28 characters

num_epochs = 200
num_hidden = 50
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-2
momentum = 0.9

num_examples = 1
num_batches_per_epoch = int(num_examples / batch_size)

imgRaw = misc.imread("ocrdata/a01-000u-s00-00.png").transpose()
print("raw image")
print(imgRaw.shape)
rawW = imgRaw.shape[0]
rawH = imgRaw.shape[1]
imgHeight = num_features
imgWidth = int(round(rawW * (imgHeight / rawH)))
imgTensor = misc.imresize(imgRaw, (imgWidth, imgHeight))

imgNdarr = np.asarray(imgTensor[np.newaxis, :])
transposedImgNdarr = imgNdarr
normalizedImgNdarr = (transposedImgNdarr - np.mean(transposedImgNdarr)) / np.std(transposedImgNdarr)

train_inputs = normalizedImgNdarr
train_seq_len = [train_inputs.shape[1]]

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
    return normalizedImgNdarr,sparse_tuple_from([label_dense]),[normalizedImgNdarr.shape[1]]

def train():
    targets_intList = np.asarray([characterListInUsage.index(x) for x in inputstring])
    print("char2index")
    print(targets_intList)

    train_targets = sparse_tuple_from([targets_intList])  # Creating sparse representation to feed the placeholder

    val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len  # We don't have a validation dataset :(

    graph = tf.Graph()
    with graph.as_default():
        inputs_sink = tf.placeholder(tf.float32, [None, None, num_features])  # num feature is input length?
        targets_intList = tf.sparse_placeholder(tf.int32)
        # 1d array of size [batch_size]
        seq_len_sink = tf.placeholder(tf.int32, [None])
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs_sink, seq_len_sink, dtype=tf.float32)

        shape = tf.shape(inputs_sink)
        batch_s, max_timesteps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        # Zero initialization
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        # Reshaping back to the original shape
        # logits = tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes])
        logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))
        loss = tf.nn.ctc_loss(targets_intList, logits, seq_len_sink)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len_sink)
        ler = tf.reduce_mean(
            tf.edit_distance(tf.cast(decoded[0], tf.int32), targets_intList))  # Inaccuracy: label error rate

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):
                feed = {inputs_sink: train_inputs,
                        targets_intList: train_targets,
                        seq_len_sink: train_seq_len}

                batch_cost, _ = sess.run([cost, optimizer], feed)
                train_cost += batch_cost * batch_size
                train_ler += sess.run(ler, feed_dict=feed) * batch_size

            train_cost /= num_examples
            train_ler /= num_examples

            val_feed = {inputs_sink: val_inputs,
                        targets_intList: val_targets,
                        seq_len_sink: val_seq_len}

            val_cost, val_ler = sess.run([cost, ler], feed_dict=val_feed)

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
            print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
                             val_cost, val_ler, time.time() - start))

        # Decoding
        result_dec = sess.run(decoded[0], feed_dict=feed)

        p(result_dec)
        p(result_dec[1])
        result_dense = result_dec[1]

        final_decoded = [characterListInUsage[i] for i in result_dense]
        print('Original:\n%s' % inputstring)
        print('Decoded:\n%s' % ''.join(final_decoded))
    # tf.saved_model.builder.SavedModelBuilder("savedModel").add_meta_graph_and_variables(session,["tfnn"])


train()
