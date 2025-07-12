import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import pathlib
import pickle

from attacks_b import AdditiveNoise1D, Butterworth1D, RandomCut
from config_b import FRAME_LEN, NUM_BITS


def _encoder():
    x_in = layers.Input((FRAME_LEN,))
    m_in = layers.Input((1, 1, NUM_BITS))

    x = tf.expand_dims(x_in, -1)  # ➜ (B,T,1)
    for nf, ks, st in [
        (16, 41, 8),
        (32, 21, 4),
        (64, 21, 2),
        (128, 21, 2),
        (256, 21, 2),
    ]:
        x = layers.Conv1D(
            nf,
            ks,
            st,
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )(x)

    # reshape latent to (B,128,256) then concat message on channel axis
    x = layers.Reshape((128, 256))(x)

    m = tf.squeeze(m_in, axis=2)  # (B,1,1,512) -> (B,1,512)
    m = tf.tile(m, [1, 128, 1])  # (B,1,512) -> (B,128,512)
    x = layers.Concatenate(axis=-1)([x, m])

    # 1×1 conv to mix message (256+512=768 channels input)
    x = layers.Conv1D(
        256, 9, padding="same", activation="selu", kernel_initializer="lecun_normal"
    )(x)

    # decoder / up-sampling
    for nf, ks, st in [(128, 21, 2), (64, 21, 2), (32, 21, 2), (16, 21, 4), (8, 21, 8)]:
        x = layers.Conv1DTranspose(
            nf,
            ks,
            st,
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )(x)

    x = layers.Conv1D(
        1, 41, padding="same", activation="tanh", kernel_initializer="glorot_uniform"
    )(x)
    out = tf.squeeze(x, -1)
    return models.Model([x_in, m_in], out, name="embedder_b")


def _attack_layer():
    taps = pickle.load(open(pathlib.Path(__file__).parent / "lpf_taps.pkl", "rb"))
    inp = layers.Input((FRAME_LEN,))
    y = Butterworth1D(taps)(inp)
    y = AdditiveNoise1D(sigma=0.02)(y)
    y = RandomCut(k=3_000)(y)
    return models.Model(inp, y, name="attacks_b")


def _detector():
    x_in = layers.Input((FRAME_LEN,))
    x = tf.expand_dims(x_in, -1)
    for nf, ks, st in [
        (32, 41, 8),
        (32, 21, 4),
        (64, 21, 2),
        (64, 11, 2),
        (128, 11, 1),
        (128, 9, 1),
    ]:
        x = layers.Conv1D(
            nf,
            ks,
            st,
            padding="same",
            activation="selu",
            kernel_initializer="lecun_normal",
        )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(NUM_BITS, activation="sigmoid")(x)
    return models.Model(x_in, x, name="detector_b")


def build_full():
    enc = _encoder()
    atk = _attack_layer()
    dec = _detector()

    s_in = layers.Input((FRAME_LEN,))
    m_in = layers.Input((1, 1, NUM_BITS))

    enc_out = enc([s_in, m_in])
    atk_out = atk(enc_out)
    dec_out = dec(atk_out)

    return models.Model(
        [s_in, m_in], [enc_out, atk_out, dec_out], name="watermark_net_b"
    )
