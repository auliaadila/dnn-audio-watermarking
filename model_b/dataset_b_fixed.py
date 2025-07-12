# model_b/dataset_b_fixed.py
"""
Fixed dataset with improved shuffle buffer and better batching
"""
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


def _wav_paths():
    """Get all wav file paths"""
    return list(Path(LIBRI_ROOT).rglob("*.wav"))


def _frame_generator():
    """Improved frame generator with better error handling"""
    paths = _wav_paths()
    print(f"Found {len(paths)} audio files in dataset")
    
    while True:
        random.shuffle(paths)
        for p in paths:
            try:
                audio, _ = librosa.load(p, sr=FS, mono=True)
                # chop into contiguous FRAME_LEN blocks
                n_frames = len(audio) // FRAME_LEN
                if n_frames == 0:
                    continue
                audio = audio[: n_frames * FRAME_LEN]
                audio = audio.reshape(-1, FRAME_LEN)
                for f in audio:
                    yield f.astype("float32")
            except Exception as e:
                print(f"Warning: Could not load {p}: {e}")
                continue


def message_batch():
    """Generate message batch (unchanged - was working correctly)"""
    # broadcast one random msg over the batch (as in paper)
    idx = random.randrange(len(MESSAGE_POOL))
    msg = MESSAGE_POOL[idx]
    return np.broadcast_to(msg, (BATCH_SIZE, NUM_BITS)).astype("float32")


def tf_dataset_fixed(shuffle_buffer_size=5000):
    """
    Fixed dataset with improved parameters
    
    Args:
        shuffle_buffer_size: Increased from 1000 to 5000 for better shuffling
    """
    print(f"Creating dataset with shuffle buffer size: {shuffle_buffer_size}")
    
    ds = tf.data.Dataset.from_generator(
        _frame_generator, output_signature=tf.TensorSpec([FRAME_LEN], tf.float32)
    )
    
    # Improved pipeline with better performance
    ds = ds.batch(BATCH_SIZE)
    ds = ds.shuffle(shuffle_buffer_size)  # Significantly increased shuffle buffer
    ds = ds.repeat()  # Prevent OutOfRange errors
    ds = ds.prefetch(tf.data.AUTOTUNE)  # Better prefetching

    def add_msg(x):
        m = tf.numpy_function(lambda _: message_batch(), [x], tf.float32)
        m.set_shape([BATCH_SIZE, NUM_BITS])
        m = tf.expand_dims(tf.expand_dims(m, 1), 2)  # shape (B,1,1,NUM_BITS)
        return (x, m)

    ds = ds.map(add_msg, num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds


def create_validation_dataset(num_batches=100):
    """Create a fixed validation dataset for consistent evaluation"""
    print(f"Creating validation dataset with {num_batches} batches")
    
    # Use a fixed seed for reproducible validation set
    original_seed = random.getstate()
    random.seed(42)
    np.random.seed(42)
    
    ds = tf_dataset_fixed(shuffle_buffer_size=1000)  # Smaller buffer for validation
    val_ds = ds.take(num_batches)
    
    # Restore original random state
    random.setstate(original_seed)
    
    return val_ds


def get_datasets_for_training():
    """Get both training and validation datasets"""
    train_ds = tf_dataset_fixed(shuffle_buffer_size=5000)
    val_ds = create_validation_dataset(num_batches=50)
    
    return train_ds, val_ds