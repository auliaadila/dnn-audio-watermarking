#!/usr/bin/env python3
"""
Simple Fixed Training Script - Addresses all identified issues

This script implements the key fixes identified during debugging:
1. Proper encoder-detector initialization training first
2. Balanced loss weights with dynamic scheduling
3. Reduced attack aggressiveness
4. Improved data pipeline
5. Better metrics

Run this instead of your original training script.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

sys.path.append("model_b")

from model_b.config_b import EPOCHS, FRAME_LEN, NUM_BITS, STEPS_PER_EPOCH
from model_b.dataset_b import tf_dataset
from model_b.models_b import _detector, _encoder, build_full


# Fixed attack parameters
class Attacks:
    """Attacks with reduced aggressiveness"""

    def __init__(self):
        # Reduced parameters based on debugging analysis
        self.noise_sigma = 0.005  # Reduced from 0.009
        self.cut_samples = 300  # Reduced from 1000

    def apply_attacks(self, x, training=False, strength=1.0):
        """Apply attacks with controllable strength"""
        if not training:
            return x

        # Noise attack (reduced intensity)
        noise = tf.random.normal(tf.shape(x), stddev=self.noise_sigma * strength)
        x = x + noise

        # Random cutting (reduced samples)
        if strength > 0.5:  # Only apply in later training
            shape = tf.shape(x)
            b, t = shape[0], shape[1]
            k = int(self.cut_samples * strength)

            if k > 0:
                idx = tf.random.uniform((b, k), 0, t, dtype=tf.int32)
                batch_idx = tf.repeat(tf.range(b)[:, None], k, 1)
                full_idx = batch_idx * t + idx

                mask = tf.scatter_nd(
                    tf.expand_dims(tf.reshape(full_idx, [-1]), -1),
                    tf.zeros(tf.shape(tf.reshape(full_idx, [-1])), dtype=x.dtype),
                    shape=(b * t,),
                )
                mask = tf.reshape(mask, (b, t))
                keep = tf.cast(mask == 0, x.dtype)  # Inverted: keep where mask is 0
                x = x * keep

        return x


def binary_accuracy_metric(y_true, y_pred):
    """Fixed binary accuracy metric"""
    predictions = tf.cast(y_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))


def snr_metric(original, modified):
    """Signal-to-noise ratio in dB"""
    signal_power = tf.reduce_mean(tf.square(original))
    noise_power = tf.reduce_mean(tf.square(original - modified))
    snr = 10.0 * tf.math.log(signal_power / (noise_power + 1e-8)) / tf.math.log(10.0)
    return snr


class FixedTrainer:
    def __init__(self):
        self.encoder = _encoder()
        self.detector = _detector()
        self.attacks = Attacks()
        self.training_history = []

        print("🔧 Fixed Trainer Initialized")
        print(f"   Encoder: {self.encoder.count_params():,} parameters")
        print(f"   Detector: {self.detector.count_params():,} parameters")
        print(f"   Attack aggressiveness: REDUCED")

    def get_improved_dataset(self):
        """Get dataset with improved shuffle buffer"""
        ds = tf_dataset()
        # Increase shuffle buffer from 1000 to 3000
        ds = ds.shuffle(3000)

        def prepare_inputs(signal, message):
            target = tf.squeeze(message, [1, 2])
            return (signal, message), target

        return ds.map(prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    def phase1_initialization_training(self, epochs=3):
        """Phase 1: Train encoder and detector together WITHOUT attacks"""
        print(f"\n{'=' * 60}")
        print("PHASE 1: ENCODER-DETECTOR INITIALIZATION")
        print(f"{'=' * 60}")
        print("Training encoder and detector together (NO ATTACKS)")
        print("This establishes a compatible embedding-detection scheme")

        # Enable training for both
        self.encoder.trainable = True
        self.detector.trainable = True

        # Create training model (encoder -> detector, no attacks)
        inp_sig = tf.keras.Input((FRAME_LEN,), name="signal")
        inp_msg = tf.keras.Input((1, 1, NUM_BITS), name="message")

        enc_out_raw = self.encoder([inp_sig, inp_msg])
        watermark_strength = 0.0008  # Reduce watermark strength to 0.08%
        enc_out = inp_sig + watermark_strength * (enc_out_raw - inp_sig)
        det_out = self.detector(enc_out)

        phase1_model = tf.keras.Model([inp_sig, inp_msg], det_out)
        phase1_model.compile(
            optimizer=tf.keras.optimizers.Nadam(1e-4),
            loss="binary_crossentropy",
            metrics=[binary_accuracy_metric],
        )

        dataset = self.get_improved_dataset()

        print(f"Training for {epochs} epochs...")
        history = phase1_model.fit(
            dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=epochs, verbose=1
        )

        final_loss = history.history["loss"][-1]
        final_acc = history.history["binary_accuracy_metric"][-1]

        print(f"\nPhase 1 Results:")
        print(f"  Loss: {final_loss:.4f}")
        print(f"  Accuracy: {final_acc:.4f} ({final_acc * 100:.1f}%)")

        success = final_loss < 0.1 and final_acc > 0.95
        print(f"  Status: {'✅ SUCCESS' if success else '❌ NEEDS MORE TRAINING'}")

        return success, history.history

    def phase2_attack_robust_training(self, epochs=8):
        """Phase 2: Train detector to be robust to attacks, encoder frozen"""
        print(f"\n{'=' * 60}")
        print("PHASE 2: ATTACK ROBUSTNESS TRAINING")
        print(f"{'=' * 60}")
        print("Training detector with attacks enabled (encoder frozen)")
        print("Using reduced attack aggressiveness")

        # Freeze encoder, train detector
        self.encoder.trainable = False
        self.detector.trainable = True

        dataset = self.get_improved_dataset()
        optimizer = tf.keras.optimizers.Nadam(2e-4)
        bce_loss = tf.keras.losses.BinaryCrossentropy()

        epoch_results = []

        for epoch in range(epochs):
            # Gradually increase attack strength
            attack_strength = (epoch + 1) / epochs
            print(
                f"\nEpoch {epoch + 1}/{epochs} - Attack strength: {attack_strength:.2f}"
            )

            epoch_losses = []
            epoch_accs = []

            for step, ((signal, message), target) in enumerate(
                dataset.take(STEPS_PER_EPOCH)
            ):
                with tf.GradientTape() as tape:
                    # Forward pass with attacks and watermark strength scaling
                    enc_out_raw = self.encoder([signal, message])
                    watermark_strength = 0.0008  # Reduce watermark strength to 0.08%
                    enc_out = signal + watermark_strength * (enc_out_raw - signal)

                    atk_out = self.attacks.apply_attacks(
                        enc_out, training=True, strength=attack_strength
                    )
                    det_out = self.detector(atk_out)

                    # Loss
                    loss = bce_loss(target, det_out)

                # Update detector only
                gradients = tape.gradient(loss, self.detector.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.detector.trainable_variables)
                )

                # Metrics
                acc = binary_accuracy_metric(target, det_out)
                epoch_losses.append(loss.numpy())
                epoch_accs.append(acc.numpy())

                if step % 200 == 0:
                    print(
                        f"  Step {step}/{STEPS_PER_EPOCH} - Loss: {loss:.4f}, Acc: {acc:.3f}"
                    )

            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accs)

            epoch_results.append(
                {
                    "epoch": epoch + 1,
                    "attack_strength": attack_strength,
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                }
            )

            print(f"  Epoch Summary - Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")

        final_result = epoch_results[-1]
        success = final_result["loss"] < 0.4 and final_result["accuracy"] > 0.7

        print(f"\nPhase 2 Results:")
        print(f"  Final Loss: {final_result['loss']:.4f}")
        print(f"  Final Accuracy: {final_result['accuracy']:.4f}")
        print(f"  Status: {'✅ SUCCESS' if success else '⚠️ PARTIAL SUCCESS'}")

        return success, epoch_results

    def phase3_end_to_end_training(self, epochs=10):
        """Phase 3: End-to-end training with proper loss balancing"""
        print(f"\n{'=' * 60}")
        print("PHASE 3: END-TO-END TRAINING")
        print(f"{'=' * 60}")
        print("Training both encoder and detector with balanced loss weights")
        print("Using dynamic weight scheduling from debugging analysis")

        # Enable training for both
        self.encoder.trainable = True
        self.detector.trainable = True

        dataset = self.get_improved_dataset()
        optimizer = tf.keras.optimizers.Nadam(1e-4)
        bce_loss = tf.keras.losses.BinaryCrossentropy()

        epoch_results = []

        for epoch in range(epochs):
            # Dynamic loss weights (from debugging recommendations)
            w_d = 3.0  # Detection weight (constant)
            w_e = 0.2 + (epoch / epochs) * 0.8  # Encoder weight (0.2 -> 1.0)

            print(
                f"\nEpoch {epoch + 1}/{epochs} - Loss weights: w_e={w_e:.2f}, w_d={w_d:.2f}"
            )

            epoch_losses = []
            epoch_accs = []
            epoch_snrs = []

            for step, ((signal, message), target) in enumerate(
                dataset.take(STEPS_PER_EPOCH)
            ):
                with tf.GradientTape() as tape:
                    # Forward pass with watermark strength scaling
                    enc_out_raw = self.encoder([signal, message])
                    watermark_strength = 0.0008  # Reduce watermark strength to 0.08%
                    enc_out = signal + watermark_strength * (enc_out_raw - signal)

                    atk_out = self.attacks.apply_attacks(
                        enc_out, training=True, strength=1.0
                    )
                    det_out = self.detector(atk_out)

                    # Combined loss
                    detection_loss = bce_loss(target, det_out)
                    encoder_loss = tf.reduce_mean(
                        tf.square(signal - enc_out)
                    )  # Signal quality
                    total_loss = w_d * detection_loss + w_e * encoder_loss

                # Update both networks
                trainable_vars = (
                    self.encoder.trainable_variables + self.detector.trainable_variables
                )
                gradients = tape.gradient(total_loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients, trainable_vars))

                # Metrics
                acc = binary_accuracy_metric(target, det_out)
                snr = snr_metric(signal, enc_out)

                epoch_losses.append(total_loss.numpy())
                epoch_accs.append(acc.numpy())
                epoch_snrs.append(snr.numpy())

                if step % 200 == 0:
                    print(
                        f"  Step {step}/{STEPS_PER_EPOCH} - Loss: {total_loss:.4f}, Acc: {acc:.3f}, SNR: {snr:.1f}dB"
                    )

            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accs)
            avg_snr = np.mean(epoch_snrs)

            epoch_results.append(
                {
                    "epoch": epoch + 1,
                    "w_e": w_e,
                    "w_d": w_d,
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                    "snr_db": avg_snr,
                }
            )

            print(
                f"  Epoch Summary - Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}, SNR: {avg_snr:.1f}dB"
            )

        final_result = epoch_results[-1]
        success = (
            final_result["accuracy"] > 0.8
            and final_result["snr_db"] > 12.0
            and final_result["loss"] < 1.5
        )

        print(f"\nPhase 3 Results:")
        print(f"  Final Loss: {final_result['loss']:.4f}")
        print(f"  Final Accuracy: {final_result['accuracy']:.4f}")
        print(f"  Final SNR: {final_result['snr_db']:.1f} dB")
        print(f"  Status: {'✅ SUCCESS' if success else '⚠️ NEEDS TUNING'}")

        return success, epoch_results

    def save_results(self, phase1_history, phase2_history, phase3_history):
        """Save all training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model weights
        weights_file = f"model_weights_fixed_{timestamp}.h5"

        # Create a temporary model to save weights
        full_model = build_full()
        full_model.get_layer("embedder_b").set_weights(self.encoder.get_weights())
        full_model.get_layer("detector_b").set_weights(self.detector.get_weights())
        full_model.save_weights(weights_file)

        # Save training history
        history_file = f"training_history_fixed_{timestamp}.json"
        history = {
            "phase1": phase1_history,
            "phase2": phase2_history,
            "phase3": phase3_history,
            "fixes_applied": {
                "proper_initialization": True,
                "reduced_attack_aggressiveness": True,
                "dynamic_loss_weights": True,
                "improved_shuffle_buffer": True,
                "fixed_accuracy_metric": True,
            },
        }

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2, default=str)

        print(f"\n💾 Results saved:")
        print(f"   Weights: {weights_file}")
        print(f"   History: {history_file}")

        return weights_file, history_file


def evaluate_ber_on_test(encoder, detector, attacks, n_test_samples=1000):
    """
    Evaluate Bit Error Rate (BER) on test set

    Args:
        encoder: Trained encoder model
        detector: Trained detector model
        attacks: Attack layer
        n_test_samples: Number of test samples to evaluate

    Returns:
        Dictionary with BER metrics
    """
    print(f"\n{'=' * 60}")
    print("📊 BER EVALUATION ON TEST SET")
    print(f"{'=' * 60}")
    print(f"Evaluating on {n_test_samples} test samples...")

    # Get test dataset (exclude train data)
    test_ds = tf_dataset(split="test")

    def prepare_test_inputs(signal, message):
        target = tf.squeeze(message, [1, 2])
        return (signal, message), target

    test_ds = test_ds.map(prepare_test_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    # Collect predictions
    total_bits = 0.0
    total_errors = 0.0
    clean_correct = 0.0
    attacked_correct = 0.0
    n_samples = 0.0

    print("Processing test samples...")
    for step, ((signal, message), target) in enumerate(test_ds.take(n_test_samples)):
        if step % 100 == 0:
            print(f"  Progress: {step}/{n_test_samples}")

        # Forward pass with watermark strength scaling
        enc_out_raw = encoder([signal, message])
        watermark_strength = 0.0008  # Reduce watermark strength to 0.08%
        enc_out = signal + watermark_strength * (enc_out_raw - signal)

        # Clean detection (no attacks)
        clean_pred = detector(enc_out)
        clean_binary = tf.cast(clean_pred > 0.5, tf.float32)

        # Attacked detection
        atk_out = attacks.apply_attacks(enc_out, training=True, strength=1.0)
        attacked_pred = detector(atk_out)
        attacked_binary = tf.cast(attacked_pred > 0.5, tf.float32)

        # Calculate errors
        batch_size = tf.shape(target)[0]
        bits_per_sample = NUM_BITS

        # Clean BER
        clean_errors = tf.reduce_sum(tf.abs(clean_binary - target))
        clean_correct += tf.reduce_sum(
            tf.cast(tf.equal(clean_binary, target), tf.float32)
        )

        # Attacked BER
        attacked_errors = tf.reduce_sum(tf.abs(attacked_binary - target))
        attacked_correct += tf.reduce_sum(
            tf.cast(tf.equal(attacked_binary, target), tf.float32)
        )

        total_errors += attacked_errors
        total_bits += tf.cast(batch_size * bits_per_sample, tf.float32)
        n_samples += tf.cast(batch_size, tf.float32)

        # Stop early if we have enough samples
        if n_samples >= n_test_samples:
            break

    # Calculate final metrics
    clean_ber = 1.0 - (clean_correct / total_bits)
    attacked_ber = 1.0 - (attacked_correct / total_bits)
    clean_accuracy = clean_correct / total_bits
    attacked_accuracy = attacked_correct / total_bits

    results = {
        "n_test_samples": int(n_samples),
        "total_bits_tested": int(total_bits),
        "clean_ber": float(clean_ber),
        "attacked_ber": float(attacked_ber),
        "clean_accuracy": float(clean_accuracy),
        "attacked_accuracy": float(attacked_accuracy),
        "test_split": "test",
    }

    print(f"\n📈 TEST RESULTS:")
    print(f"   Samples evaluated: {n_samples}")
    print(f"   Total bits tested: {total_bits:,}")
    print(f"   Clean BER: {clean_ber:.4f} ({clean_ber * 100:.2f}%)")
    print(f"   Attacked BER: {attacked_ber:.4f} ({attacked_ber * 100:.2f}%)")
    print(f"   Clean Accuracy: {clean_accuracy:.4f} ({clean_accuracy * 100:.1f}%)")
    print(
        f"   Attacked Accuracy: {attacked_accuracy:.4f} ({attacked_accuracy * 100:.1f}%)"
    )

    # Performance assessment
    if attacked_ber < 0.1:
        print("   ✅ EXCELLENT: BER < 10%")
    elif attacked_ber < 0.2:
        print("   ✅ GOOD: BER < 20%")
    elif attacked_ber < 0.3:
        print("   ⚠️ ACCEPTABLE: BER < 30%")
    else:
        print("   ❌ POOR: BER > 30%")

    return results


def main():
    """Main training function"""
    print("🚀 FIXED DNN AUDIO WATERMARKING TRAINING")
    print("   Implementing all fixes from debugging analysis")
    print(f"   Timestamp: {datetime.now()}")

    trainer = FixedTrainer()

    # Phase 1: Initialization
    success1, history1 = trainer.phase1_initialization_training(epochs=2)
    if not success1:
        print("\n❌ Phase 1 failed. Training stopped.")
        print("   Try increasing epochs or adjusting learning rate.")
        return False

    # Phase 2: Attack robustness
    success2, history2 = trainer.phase2_attack_robust_training(epochs=3)

    # Phase 3: End-to-end (proceed regardless of Phase 2 result)
    success3, history3 = trainer.phase3_end_to_end_training(epochs=3)

    # Save results
    weights_file, history_file = trainer.save_results(history1, history2, history3)

    # Evaluate BER on test set
    print("\n🔍 Evaluating BER on test set...")
    test_results = evaluate_ber_on_test(
        trainer.encoder, trainer.detector, trainer.attacks, n_test_samples=1000
    )

    # Save test results
    test_results_file = (
        f"test_results_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(test_results_file, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"   Test results saved: {test_results_file}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("🎯 TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Phase 1 (Initialization):     {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"Phase 2 (Attack Robustness):  {'✅ SUCCESS' if success2 else '⚠️ PARTIAL'}")
    print(
        f"Phase 3 (End-to-End):         {'✅ SUCCESS' if success3 else '⚠️ NEEDS TUNING'}"
    )

    overall_success = success1 and (success2 or success3)
    if overall_success:
        print("\n🎉 TRAINING SUCCESSFUL!")
        print("   Your watermarking system is ready.")
        print(f"   Load weights from: {weights_file}")
    else:
        print("\n⚠️ TRAINING PARTIALLY SUCCESSFUL")
        print("   System functional but may need tuning.")

    return overall_success


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    success = main()
