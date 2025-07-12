# model_b/dataset_b.py
import tensorflow as tf
import librosa
import random
from pathlib import Path
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config_b import FS, FRAME_LEN, BATCH_SIZE, LIBRI_ROOT, MESSAGE_POOL, NUM_BITS


def _wav_paths(split="train"):
    """Get wav paths for specific split (train/val/test)"""
    if split == "all":
        return list(Path(LIBRI_ROOT).rglob("*.wav"))
    else:
        return list(Path(LIBRI_ROOT).glob(f"{split}/*.wav"))


def _frame_generator(split="train"):
    paths = _wav_paths(split)
    while True:
        random.shuffle(paths)
        for p in paths:
            audio, _ = librosa.load(p, sr=FS, mono=True)
            # chop into contiguous FRAME_LEN blocks
            n_frames = len(audio) // FRAME_LEN
            if n_frames == 0:
                continue
            audio = audio[: n_frames * FRAME_LEN]
            audio = audio.reshape(-1, FRAME_LEN)
            for f in audio:
                yield f.astype("float32")


def message_batch():
    # broadcast one random msg over the batch (as in paper)
    idx = random.randrange(len(MESSAGE_POOL))
    msg = MESSAGE_POOL[idx]
    return np.broadcast_to(msg, (BATCH_SIZE, NUM_BITS)).astype("float32")


def tf_dataset(split="train"):
    ds = tf.data.Dataset.from_generator(
        lambda: _frame_generator(split), output_signature=tf.TensorSpec([FRAME_LEN], tf.float32)
    )
    # Optimize for faster training
    ds = ds.batch(BATCH_SIZE)
    ds = ds.shuffle(1000)  # Smaller shuffle buffer 
    ds = ds.repeat()  # Prevent OutOfRange errors
    ds = ds.prefetch(tf.data.AUTOTUNE)

    def add_msg(x):
        m = tf.numpy_function(lambda _: message_batch(), [x], tf.float32)
        m.set_shape([BATCH_SIZE, NUM_BITS])
        m = tf.expand_dims(tf.expand_dims(m, 1), 2)  # shape (B,1,1,NUM_BITS)
        return (x, m)

    return ds.map(add_msg, num_parallel_calls=tf.data.AUTOTUNE)