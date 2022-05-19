import sys

import tensorflow as tf
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def call(self, x):
        # linear = tf.keras.Sequential()
        dense = tf.keras.layers.Dense(1, activation='tanh')
        # linear.add(tf.keras.layers.Dropout(.2, ))
        # linear.add(tf.keras.layers.Dense(1, activation=None))
        attention_out = dense(x)
        weight = tf.nn.softmax(attention_out, dim=1)
        print(tf.transpose(x,perm = [1, 0]).shape,weight.shape)
        ret = tf.matmul(tf.transpose(x,perm = [1, 0]), weight).squeeze(2)
        print(ret.shape)
        sys.exit(0)
        return ret