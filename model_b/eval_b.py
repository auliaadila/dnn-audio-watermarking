# model_b/eval_b.py
import tensorflow as tf, numpy as np, librosa, pathlib, sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from models_b import build_full
from config_b import FS, FRAME_LEN, NUM_BITS
from utils import snr
from attacks_b import AdditiveNoise1D  # optional online attack layer

# ------------------------------------------------------------------ #
def load_embedder(path):
    full = tf.keras.models.load_model(path)
    return full.get_layer("embedder_b")


def load_detector(path):
    full = tf.keras.models.load_model(path)
    return full.get_layer("detector_b")


# ------------------------------------------------------------------ #
def frame_audio(wav):
    pads = (FRAME_LEN - len(wav) % FRAME_LEN) % FRAME_LEN
    wav = np.pad(wav, (0, pads))
    return wav.reshape(-1, FRAME_LEN)


def embed(embedder, audio, message):
    frames = frame_audio(audio)
    message = np.broadcast_to(message, (len(frames), NUM_BITS))
    message = message[:, None, None, :]
    water = embedder([frames, message])
    return water.flatten()[: len(audio)]  # trim possible pad


def detect(detector, audio):
    frames = frame_audio(audio)
    out = detector.predict(frames, verbose=0)
    return (out >= 0.5).astype(int).flatten()


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python eval_b.py <embedder_dir> <detector_dir> "
              "<wav_path>")
        sys.exit(0)

    emb = load_embedder(sys.argv[1])
    det = load_detector(sys.argv[2])
    wav, _ = librosa.load(sys.argv[3], sr=FS, mono=True)

    msg = np.random.randint(0, 2, NUM_BITS).astype("float32")
    wm = embed(emb, wav, msg)

    print("SNR(dB):", snr(wav, wm))
    est = detect(det, wm)
    ber = np.mean(est != msg)
    print("BER  :", ber)