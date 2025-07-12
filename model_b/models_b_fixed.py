# model_b/models_b_fixed.py
"""
Fixed model definitions using improved attack layers
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from attacks_b_fixed import AdditiveNoise1D, Butterworth1D, RandomCut
from config_b import FRAME_LEN, NUM_BITS
import pickle, pathlib

SIG_LEN = FRAME_LEN  # convenience


def _encoder():
    """Encoder architecture (unchanged - was working correctly)"""
    x_in = layers.Input((SIG_LEN,))
    m_in = layers.Input((1, 1, NUM_BITS))

    x = tf.expand_dims(x_in, -1)                # ➜ (B,T,1)
    for nf, ks, st in [(16, 41, 8), (32, 21, 4), (64, 21, 2),
                       (128, 21, 2), (256, 21, 2)]:
        x = layers.Conv1D(nf, ks, st, padding="same",
                          activation="selu",
                          kernel_initializer="lecun_normal")(x)

    # reshape latent to (B,128,256) then concat message on channel axis
    x = layers.Reshape((128, 256))(x)
    m = tf.squeeze(m_in, axis=2)  # (B,1,1,128) -> (B,1,128)
    m = tf.tile(m, [1, 128, 1])      # (B,1,128) -> (B,128,128)
    x = layers.Concatenate(axis=-1)([x, m])

    # 1×1 conv to mix message (256+128=384 channels input)
    x = layers.Conv1D(256, 9, padding="same",
                      activation="selu",
                      kernel_initializer="lecun_normal")(x)

    # decoder / up-sampling
    for nf, ks, st in [(128, 21, 2), (64, 21, 2), (32, 21, 2),
                       (16, 21, 4), (8, 21, 8)]:
        x = layers.Conv1DTranspose(nf, ks, st,
                                   padding="same",
                                   activation="selu",
                                   kernel_initializer="lecun_normal")(x)

    x = layers.Conv1D(1, 41, padding="same",
                      activation="tanh",
                      kernel_initializer="glorot_uniform")(x)
    out = tf.squeeze(x, -1)
    return models.Model([x_in, m_in], out, name="embedder_b")


def _attack_layer_fixed():
    """Fixed attack layer with reduced aggressiveness"""
    taps = pickle.load(open(pathlib.Path(__file__).parent /
                            "lpf_taps.pkl", "rb"))
    inp = layers.Input((SIG_LEN,))
    
    # Apply fixed attacks with reduced parameters
    y = Butterworth1D(taps)(inp)  # Filter attack
    y = AdditiveNoise1D(sigma=0.005)(y)  # Reduced noise (was 0.009)
    y = RandomCut(k=300)(y)  # Reduced cutting (was 1000)
    
    return models.Model(inp, y, name="attacks_b_fixed")


def _detector():
    """Detector architecture (unchanged - was working correctly)"""
    x_in = layers.Input((SIG_LEN,))
    x = tf.expand_dims(x_in, -1)
    for nf, ks, st in [(32, 41, 8), (32, 21, 4), (64, 21, 2),
                       (64, 11, 2), (128, 11, 1), (128, 9, 1)]:
        x = layers.Conv1D(nf, ks, st,
                          padding="same",
                          activation="selu",
                          kernel_initializer="lecun_normal")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(NUM_BITS, activation="sigmoid")(x)
    return models.Model(x_in, x, name="detector_b")


def build_full_fixed():
    """Build full model with fixed attack parameters"""
    enc = _encoder()
    atk = _attack_layer_fixed()
    dec = _detector()

    s_in = layers.Input((SIG_LEN,))
    m_in = layers.Input((1, 1, NUM_BITS))

    enc_out = enc([s_in, m_in])
    atk_out = atk(enc_out)
    dec_out = dec(atk_out)

    return models.Model([s_in, m_in], [enc_out, atk_out, dec_out],
                        name="watermark_net_b_fixed")


def build_progressive_model():
    """Build model for progressive training (with gradual attack introduction)"""
    enc = _encoder()
    dec = _detector()
    
    # Load taps for gradual attacks
    taps_path = pathlib.Path(__file__).parent / "lpf_taps.pkl"
    
    s_in = layers.Input((SIG_LEN,))
    m_in = layers.Input((1, 1, NUM_BITS))

    enc_out = enc([s_in, m_in])
    
    # Create a custom layer for progressive attacks
    class ProgressiveAttacks(layers.Layer):
        def __init__(self, taps_path, **kwargs):
            super().__init__(**kwargs)
            import pickle
            self.taps = pickle.load(open(taps_path, "rb"))
            self.noise_layer = AdditiveNoise1D(sigma=0.005)
            self.filter_layer = Butterworth1D(self.taps)
            self.cut_layer = RandomCut(k=300)
            
        def call(self, inputs, training=False, attack_strength=1.0):
            if not training:
                return inputs
                
            x = inputs
            
            # Apply attacks with controllable strength
            # Noise (scale sigma)
            if attack_strength > 0:
                noise = tf.random.normal(tf.shape(x), stddev=0.005 * attack_strength)
                x = x + noise
                
                # Filter (interpolate)
                filtered = self.filter_layer(x, training=True)
                x = (1.0 - attack_strength) * x + attack_strength * filtered
                
                # Cut (scale number of cuts)
                if attack_strength > 0.5:  # Only apply cutting in later stages
                    x = self.cut_layer(x, training=True)
            
            return x
    
    prog_attacks = ProgressiveAttacks(taps_path, name="progressive_attacks")
    dec_out = dec(enc_out)  # We'll add attacks later in training script

    return models.Model([s_in, m_in], [enc_out, dec_out],
                        name="progressive_watermark_net_b"), prog_attacks