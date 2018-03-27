import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
from six.moves import xrange as range
from python_speech_features import mfcc
from utils import sparse_tuple_from as sparse_tuple_from

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

fs, audio = wav.read('LDC93S1.wav')
inputs = mfcc(audio, samplerate=fs)

train_inputs = np.asarray(inputs[np.newaxis, :])
train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
train_seq_len = [train_inputs.shape[1]]
with open('LDC93S1.txt', 'r') as f:
    line = f.readlines()[-1]#Only the last line is necessary
    original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')# Get only the words between [a-z] and replace period for none
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')

targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])# Adding blank label
targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])# Transform char into index
train_targets = sparse_tuple_from([targets])# Creating sparse representation to feed the placeholder
val_inputs, val_targets, val_seq_len = train_inputs, train_targets, train_seq_len# We don't have a validation dataset :(

with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, ["tfnn"], "savedModel")
  seq_len = tf.placeholder(tf.int32, [None])

  feed = {inputs: train_inputs,
          targets: train_targets,
          seq_len: train_seq_len}

  result_dec = sess.run(decoded[0], feed_dict=feed)

  str_decoded = ''.join([chr(x) for x in np.asarray(result_dec[1]) + FIRST_INDEX])
  # Replacing blank label to none
  str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
  # Replacing space label to space
  str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
