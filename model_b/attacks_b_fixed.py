# model_b/attacks_b_fixed.py
"""
Fixed attack layers with reduced aggressiveness based on debugging analysis
"""
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
    """Reduced noise intensity from debugging analysis"""
    def __init__(self, sigma=0.005, **kw):  # Reduced from 0.009 to 0.005
        super().__init__(**kw)
        self.sigma = sigma

    def call(self, x, training=False):
        if not training:
            return x
        noise = tf.random.normal(tf.shape(x), stddev=self.sigma)
        return x + noise


class Butterworth1D(Layer):
    """FIR implementation with higher cutoff frequency for less destructive filtering"""
    def __init__(self, taps, **kw):
        super().__init__(**kw)
        # Note: Consider generating new taps with higher cutoff (6-8kHz instead of 4kHz)
        # For now, using existing taps but documenting the issue
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
    """Reduced sample cutting from debugging analysis"""
    def __init__(self, k=300, **kw):  # Reduced from 1000 to 300 (~0.9% instead of 3%)
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


class GradualAttacks(Layer):
    """New layer for gradual attack introduction during training"""
    def __init__(self, taps_path, **kw):
        super().__init__(**kw)
        # Load filter taps
        import pickle
        self.taps = pickle.load(open(taps_path, "rb"))
        
        # Create individual attack layers
        self.noise_layer = AdditiveNoise1D(sigma=0.005)
        self.filter_layer = Butterworth1D(self.taps)
        self.cut_layer = RandomCut(k=300)
        
        # Attack strength parameter (will be set externally)
        self.attack_strength = tf.Variable(1.0, trainable=False, name='attack_strength')
    
    def set_attack_strength(self, strength):
        """Set the strength of attacks (0.0 = no attacks, 1.0 = full attacks)"""
        self.attack_strength.assign(strength)
    
    def call(self, x, training=False):
        if not training:
            return x
            
        # Apply attacks with varying intensity
        y = x
        
        # Apply each attack with the current strength
        if self.attack_strength > 0.0:
            # Noise attack (scale sigma with strength)
            noise_sigma = 0.005 * self.attack_strength
            noise = tf.random.normal(tf.shape(y), stddev=noise_sigma)
            y = y + noise
            
            # Filter attack (interpolate between original and filtered)
            filtered = self.filter_layer(y, training=True)
            y = (1.0 - self.attack_strength) * y + self.attack_strength * filtered
            
            # Cut attack (scale number of cuts with strength)
            cut_k = tf.cast(300 * self.attack_strength, tf.int32)
            if cut_k > 0:
                shape = tf.shape(y)
                b = shape[0]
                idx = tf.random.uniform((b, cut_k), 0, FRAME_LEN, dtype=tf.int32)
                batch = tf.repeat(tf.range(b)[:, None], cut_k, 1)
                full = tf.cast(batch * FRAME_LEN + idx, tf.int64)
                mask = tf.scatter_nd(
                    tf.expand_dims(full, -1),
                    tf.zeros_like(full, dtype=y.dtype),
                    shape=(b * FRAME_LEN,),
                )
                mask = tf.reshape(mask, (b, FRAME_LEN))
                keep = tf.cast(mask != 0, y.dtype)
                y = y * keep
        
        return y