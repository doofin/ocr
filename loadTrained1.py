from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
from six.moves import xrange as range
from python_speech_features import mfcc
from utils import sparse_tuple_from as sparse_tuple_from

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
num_features = 13
num_classes = ord('z') - ord('a') + 1 + 1 + 1# Accounting the 0th indice +  space + blank label = 28 characters

num_epochs = 70
num_hidden = 50
num_layers = 1
batch_size = 1
initial_learning_rate = 1e-2
momentum = 0.9

num_examples = 1
num_batches_per_epoch = int(num_examples/batch_size)

# trainInputFn='LDC93S1.wav'
# trainLableFn='LDC93S1.txt'

trainInputFn='./dataset/LDC93S1.wav'
trainLableFn='./dataset/LDC93S1.txt'

fs, audio = wav.read(trainInputFn)
inputs = mfcc(audio, samplerate=fs)
# Tranform in 3D array
train_inputs = np.asarray(inputs[np.newaxis, :])
train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
train_seq_len = [train_inputs.shape[1]]

with open(trainLableFn, 'r') as f:
    line = f.readlines()[-1]#Only the last line is necessary
    original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')# Get only the words between [a-z] and replace period for none
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')

targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])# Adding blank label
targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])# Transform char into index
train_targets = sparse_tuple_from([targets])# Creating sparse representation to feed the placeholder

val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len# We don't have a validation dataset :(

graph = tf.Graph()
with graph.as_default():
    inputs = tf.placeholder(tf.float32, [None, None, num_features])
    # Here we use sparse_placeholder that will generate a SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)
    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
    # Zero initialization
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    # Reshaping back to the original shape
    # logits = tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes])
    logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [batch_s, -1, num_classes]), (1, 0, 2))
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,0.9).minimize(cost)
    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))# Inaccuracy: label error rate

with tf.Session(graph=graph) as session:
    # tf.global_variables_initializer().run()
    # for curr_epoch in range(num_epochs):
    #     train_cost = train_ler = 0
    #     start = time.time()
    #
    #     for batch in range(num_batches_per_epoch):
    #
    #         feed = {inputs: train_inputs,
    #                 targets: train_targets,
    #                 seq_len: train_seq_len}
    #
    #         batch_cost, _ = session.run([cost, optimizer], feed)
    #         train_cost += batch_cost*batch_size
    #         train_ler += session.run(ler, feed_dict=feed)*batch_size
    #
    #     train_cost /= num_examples
    #     train_ler /= num_examples
    #
    #     val_feed = {inputs: val_inputs,
    #                 targets: val_targets,
    #                 seq_len: val_seq_len}
    #
    #     val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)
    #
    #     log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
    #     print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
    #                      val_cost, val_ler, time.time() - start))

    tf.saved_model.loader.load(session, ["tfnn"], "savedModel")
    # Decoding
    # result_dec = session.run(decoded[0], feed_dict=feed)

    result_dec = session.run(decoded[0])
    str_decoded = ''.join([chr(x) for x in np.asarray(result_dec[1]) + FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')


    print('Original:\n%s' % original)
    print('Decoded:\n%s' % str_decoded)
    tf.saved_model.builder.SavedModelBuilder("savedModel").add_meta_graph_and_variables(session,["tfnn"])
