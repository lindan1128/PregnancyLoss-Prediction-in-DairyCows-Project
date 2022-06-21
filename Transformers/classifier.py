__author__ = "Dan LIN"

import os

import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from keras import backend as K

class T2V(Layer):
    """ Time2Vec
    Adapted from the https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    """ 
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                shape=(1, self.output_dim),
                                initializer='uniform',
                                trainable=True)
        self.P = self.add_weight(name='P',
                                shape=(1, self.output_dim),
                                initializer='uniform',
                                trainable=True)
        self.w = self.add_weight(name='w',
                                shape=(1, 1),
                                initializer='uniform',
                                trainable=True)
        self.p = self.add_weight(name='p',
                                shape=(1, 1),
                                initializer='uniform',
                                trainable=True)
        super(T2V, self).build(input_shape)
        
    def call(self, x):
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        return K.concatenate([sin_trans, original], -1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'output_dim': self.output_dim})
        return config

@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# from keras_contrib import InstanceNormalization
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.9):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x, w = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x, return_attention_scores=True)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res, w