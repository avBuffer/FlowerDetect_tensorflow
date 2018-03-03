# coding: utf-8

"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import tensorflow as tf

flags = tf.app.flags

############################
#    environment setting   #
############################
flags.DEFINE_string('data_path', '../data/flowers/', 'path for data set')
flags.DEFINE_string('model_path', '../checkpoints/', 'path for save models')

flags.DEFINE_string('test_path', '../test/', 'path for test')
flags.DEFINE_string('result_path', '../result/', 'path for save result')

############################
#    hyper parameters      #
############################
flags.DEFINE_integer('width', 100, 'image shape width')
flags.DEFINE_integer('height', 100, 'image shape height')
flags.DEFINE_integer('channel', 3, 'image channel')

flags.DEFINE_integer('batch_size', 64, 'get data batch size')

flags.DEFINE_float('train_ratio', 0.8, 'validated data ratio')
flags.DEFINE_float('learn_rate', 0.001, 'learning rate')

flags.DEFINE_integer('epochs', 500, 'train epochs')

flags.DEFINE_integer('val_freq', 50, 'the frequency of saving valid (step)')
flags.DEFINE_integer('save_freq', 100, 'the frequency of saving model(epoch)')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
