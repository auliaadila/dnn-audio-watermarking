# model_b/config_b.py
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
FS = 16_000  # 16 kHz sampling rate
FRAME_LEN = 32_768  # 2.048 s
BATCH_SIZE = 64
LIBRI_ROOT = Path("dataset/Libri2h")  # √✓ change here if needed

# ----------------------------------------------------------------------
# Watermark
# ----------------------------------------------------------------------
NUM_BITS = 512  # Increased to match Model A payload capacity
MESSAGE_POOL = np.load("samples/message_pool.npy")[:, :NUM_BITS]

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
EPOCHS = 50
STEPS_PER_EPOCH = 2_000          # ≈ two hours / epoch with BATCH_SIZE=64
LR = 2e-4