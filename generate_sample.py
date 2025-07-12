#!/usr/bin/env python3
"""
Generate a single sample with original and watermarked audio files

This script:
1. Loads the trained model
2. Takes a single audio sample from the dataset
3. Generates a watermark message
4. Creates watermarked audio
5. Saves both original.wav and watermarked.wav
6. Verifies the watermark can be detected
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

sys.path.append("/home/agung.sorlawan/dnn-audio-watermarking")
sys.path.append("/home/agung.sorlawan/dnn-audio-watermarking/model_b")

from model_b.config_b import FRAME_LEN, FS, MESSAGE_POOL, NUM_BITS
from model_b.dataset_b import tf_dataset
from model_b.models_b import build_full


def binary_accuracy_metric(y_true, y_pred):
    """Fixed binary accuracy metric"""
    predictions = tf.cast(y_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))


def load_trained_model(weights_path):
    """Load the trained model weights"""
    print(f"Loading trained model from: {weights_path}")

    # Build the full model
    model = build_full()

    # Load weights
    model.load_weights(weights_path)

    print("✅ Model loaded successfully")
    return model


def generate_sample_files(model, output_dir="sample_output"):
    """Generate original.wav and watermarked.wav files"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🎵 Generating sample files in: {output_dir}")

    # Get a single sample from the dataset
    print("Getting sample from dataset...")
    ds = tf_dataset(split="train").take(1)

    for signal, message in ds:
        # Take just the first sample from the batch
        original_signal = signal[0:1]  # Keep batch dimension
        message_bits = message[0:1]  # Keep batch dimension

        print(f"   Original signal shape: {original_signal.shape}")
        print(f"   Message shape: {message_bits.shape}")
        print(f"   Message bits: {NUM_BITS}")

        # Generate watermarked signal with reduced strength
        print("Generating watermarked signal...")
        encoder_out_raw, attack_out, detector_out = model(
            [original_signal, message_bits]
        )

        # Apply watermark strength scaling for better audio quality
        watermark_strength = 0.0008  # Reduce watermark strength to 0.08% (match training)
        encoder_out = original_signal + watermark_strength * (
            encoder_out_raw - original_signal
        )

        # Generate attacked audio from the scaled watermarked signal
        attack_out_scaled = model.get_layer("attacks_b")(encoder_out)

        # Convert to numpy for saving
        original_audio = original_signal.numpy().flatten()
        watermarked_audio = encoder_out.numpy().flatten()
        attacked_audio = attack_out_scaled.numpy().flatten()

        # Save audio files
        original_path = os.path.join(output_dir, "original.wav")
        watermarked_path = os.path.join(output_dir, "watermarked.wav")
        attacked_path = os.path.join(output_dir, "attacked.wav")

        sf.write(original_path, original_audio, FS)
        sf.write(watermarked_path, watermarked_audio, FS)
        sf.write(attacked_path, attacked_audio, FS)

        print(f"   ✅ Saved: {original_path}")
        print(f"   ✅ Saved: {watermarked_path}")
        print(f"   ✅ Saved: {attacked_path}")

        # Verify watermark detection
        print(f"\n🔍 Verifying watermark detection...")

        # Test detection on watermarked (clean) audio
        clean_detection = model.get_layer("detector_b")(encoder_out)
        clean_accuracy = binary_accuracy_metric(
            tf.squeeze(message_bits, [1, 2]), clean_detection
        )

        # Test detection on attacked audio (need to recompute with proper strength)
        attack_out_scaled = model.get_layer("attacks_b")(encoder_out)
        attacked_detection = model.get_layer("detector_b")(attack_out_scaled)
        attacked_accuracy = binary_accuracy_metric(
            tf.squeeze(message_bits, [1, 2]), attacked_detection
        )

        print(
            f"   Clean detection accuracy: {clean_accuracy:.4f} ({clean_accuracy * 100:.1f}%)"
        )
        print(
            f"   Attacked detection accuracy: {attacked_accuracy:.4f} ({attacked_accuracy * 100:.1f}%)"
        )

        # Show actual vs predicted bits (first 16 bits for readability)
        original_bits = tf.squeeze(message_bits, [1, 2]).numpy()[0]
        clean_pred_bits = (clean_detection.numpy()[0] > 0.5).astype(int)
        attacked_pred_bits = (attacked_detection.numpy()[0] > 0.5).astype(int)

        print(f"\n📊 Bit comparison (first 16 bits):")
        print(f"   Original:  {original_bits[:16].astype(int)}")
        print(f"   Clean:     {clean_pred_bits[:16]}")
        print(f"   Attacked:  {attacked_pred_bits[:16]}")

        # Calculate SNR
        signal_power = np.mean(original_audio**2)
        noise_power = np.mean((original_audio - watermarked_audio) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-8))

        print(f"\n📈 Audio Quality:")
        print(f"   SNR: {snr_db:.1f} dB")
        print(f"   Original RMS: {np.sqrt(np.mean(original_audio**2)):.4f}")
        print(f"   Watermarked RMS: {np.sqrt(np.mean(watermarked_audio**2)):.4f}")
        print(f"   Attacked RMS: {np.sqrt(np.mean(attacked_audio**2)):.4f}")

        # Save metadata
        metadata = {
            "message_bits": NUM_BITS,
            "sample_rate": FS,
            "duration_seconds": len(original_audio) / FS,
            "snr_db": float(snr_db),
            "clean_detection_accuracy": float(clean_accuracy),
            "attacked_detection_accuracy": float(attacked_accuracy),
            "original_message": original_bits.astype(int).tolist(),
            "clean_predicted": clean_pred_bits.tolist(),
            "attacked_predicted": attacked_pred_bits.tolist(),
        }

        import json

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✅ Saved: {metadata_path}")

        return {
            "original_path": original_path,
            "watermarked_path": watermarked_path,
            "attacked_path": attacked_path,
            "metadata_path": metadata_path,
            "clean_accuracy": float(clean_accuracy),
            "attacked_accuracy": float(attacked_accuracy),
            "snr_db": float(snr_db),
        }


def main():
    """Main function"""
    print("🎯 GENERATING WATERMARKED AUDIO SAMPLE")
    print("=" * 60)

    # Find the most recent model weights
    weights_files = list(Path(".").glob("model_weights_fixed_*.h5"))
    if not weights_files:
        print("❌ No trained model weights found!")
        print("   Please run training first to generate model weights.")
        return False

    # Get the most recent weights file
    latest_weights = max(weights_files, key=lambda x: x.stat().st_mtime)
    print(f"Using weights: {latest_weights}")

    # Load model
    model = load_trained_model(latest_weights)

    # Generate sample files
    results = generate_sample_files(model)

    print(f"\n{'=' * 60}")
    print("🎉 SAMPLE GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Files generated:")
    print(f"   📁 {results['original_path']}")
    print(f"   📁 {results['watermarked_path']}")
    print(f"   📁 {results['attacked_path']}")
    print(f"   📁 {results['metadata_path']}")

    print(f"\nPerformance:")
    print(f"   Clean Detection: {results['clean_accuracy'] * 100:.1f}%")
    print(f"   Attacked Detection: {results['attacked_accuracy'] * 100:.1f}%")
    print(f"   SNR: {results['snr_db']:.1f} dB")

    if results["clean_accuracy"] > 0.95 and results["attacked_accuracy"] > 0.8:
        print("   ✅ EXCELLENT watermark quality!")
    elif results["clean_accuracy"] > 0.9 and results["attacked_accuracy"] > 0.7:
        print("   ✅ GOOD watermark quality!")
    else:
        print("   ⚠️ Watermark quality needs improvement")

    return True


if __name__ == "__main__":
    # Set up GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    success = main()

