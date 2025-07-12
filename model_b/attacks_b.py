# model_b/attacks_b.py
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D
from tensorflow import keras
from config_b import FRAME_LEN
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import bwh  # already available

class AdditiveNoise1D(Layer):
    def __init__(self, sigma=0.02, **kw):
        super().__init__(**kw)
        self.sigma = sigma

    def call(self, x, training=False):
        if not training:
            return x
        noise = tf.random.normal(tf.shape(x), stddev=self.sigma)
        return x + noise


class Butterworth1D(Layer):
    """FIR implementation (weights frozen) of 16-th order 4 kHz LPF."""
    def __init__(self, taps, **kw):
        super().__init__(**kw)
        self.fir = Conv1D(
            1,
            kernel_size=len(taps),
            padding="same",
            use_bias=False,
            trainable=False,
            kernel_initializer=keras.initializers.Constant(taps[::-1][:, None]),
        )

    def call(self, x, training=False):
        if not training:
            return x
        # [B, T] ➜ [B, T, 1] ➜ conv ➜ squeeze
        y = self.fir(tf.expand_dims(x, -1))
        return tf.squeeze(y, -1)


class RandomCut(Layer):
    """Set K random samples to 0."""
    def __init__(self, k=3_000, **kw):
        super().__init__(**kw)
        self.k = k

    def call(self, x, training=False):
        if not training:
            return x
        shape = tf.shape(x)
        b = shape[0]
        idx = tf.random.uniform((b, self.k), 0, FRAME_LEN, dtype=tf.int32)
        batch = tf.repeat(tf.range(b)[:, None], self.k, 1)
        full = tf.cast(batch * FRAME_LEN + idx, tf.int64)
        mask = tf.scatter_nd(
            tf.expand_dims(full, -1),
            tf.zeros_like(full, dtype=x.dtype),
            shape=(b * FRAME_LEN,),
        )
        mask = tf.reshape(mask, (b, FRAME_LEN))
        keep = tf.cast(mask != 0, x.dtype)
        return x * keep