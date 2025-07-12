#!/usr/bin/env python3
"""
Fixed training script for Model B that addresses all identified issues:

1. Proper encoder-detector initialization training
2. Balanced loss weights with dynamic scheduling
3. Gradual attack introduction
4. Reduced attack aggressiveness
5. Improved data pipeline
6. Better metrics and monitoring
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import tensorflow as tf
import numpy as np
from dataset_b import tf_dataset
from config_b import FRAME_LEN, NUM_BITS, EPOCHS, STEPS_PER_EPOCH
from models_b import build_full
import matplotlib.pyplot as plt

# Fix for better metrics
def binary_accuracy_metric(y_true, y_pred):
    """Custom binary accuracy metric for watermark detection"""
    predictions = tf.cast(y_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))

def snr_metric(y_true, y_pred):
    """Signal-to-noise ratio metric"""
    noise_power = tf.reduce_mean(tf.square(y_true - y_pred))
    signal_power = tf.reduce_mean(tf.square(y_true))
    snr = 10.0 * tf.math.log(signal_power / (noise_power + 1e-8)) / tf.math.log(10.0)
    return snr

class ProgressiveTraining:
    """Progressive training strategy that addresses all identified issues"""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.detector = None
        self.attacks = None
        self.training_history = []
        
    def build_model(self):
        """Build model with fixed components"""
        print("Building model with fixed attack parameters...")
        
        # Build full model
        self.model = build_full()
        
        # Get components
        self.encoder = self.model.get_layer('embedder_b')
        self.detector = self.model.get_layer('detector_b')
        self.attacks = self.model.get_layer('attacks_b')
        
        print(f"✓ Encoder: {self.encoder.count_params():,} parameters")
        print(f"✓ Detector: {self.detector.count_params():,} parameters")
        print(f"✓ Attacks: {self.attacks.count_params():,} parameters")
        
    def get_improved_dataset(self):
        """Get dataset with improved shuffle buffer"""
        print("Creating improved dataset with larger shuffle buffer...")
        
        # Get base dataset
        ds = tf_dataset()
        
        # Increase shuffle buffer significantly (from 1000 to 5000)
        ds = ds.shuffle(5000)
        
        # Map to proper input format
        def prepare_inputs(signal, message):
            target = tf.squeeze(message, [1, 2])  # (batch, NUM_BITS)
            return [signal, message], target
            
        ds = ds.map(prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
        
    def phase1_encoder_detector_training(self, epochs=3):
        """Phase 1: Train encoder and detector together without attacks"""
        print(f"\n{'='*60}")
        print("PHASE 1: ENCODER-DETECTOR INITIALIZATION TRAINING")
        print(f"{'='*60}")
        print("Training encoder and detector together WITHOUT attacks")
        print("This establishes a compatible embedding-detection scheme")
        
        # Create model without attacks
        inp_sig = tf.keras.Input((FRAME_LEN,), name='input_signal')
        inp_msg = tf.keras.Input((1, 1, NUM_BITS), name='input_message')
        
        # Encoder -> Detector (no attacks)
        enc_out = self.encoder([inp_sig, inp_msg])
        det_out = self.detector(enc_out)
        
        phase1_model = tf.keras.Model([inp_sig, inp_msg], [enc_out, det_out], name='phase1_model')
        
        # Enable training for both encoder and detector
        self.encoder.trainable = True
        self.detector.trainable = True
        self.attacks.trainable = False
        
        # Compile with balanced losses
        def combined_loss(y_true, y_pred):
            enc_out, det_out = y_pred[0], y_pred[1]
            orig_sig = y_true  # This should be the target message, we'll fix this
            
            # Detector loss (main objective)
            det_loss = tf.keras.losses.BinaryCrossentropy()(y_true, det_out)
            
            # Encoder loss (perceptual quality) - very small weight initially
            # We want minimal signal distortion
            enc_loss = tf.reduce_mean(tf.square(inp_sig - enc_out))  # MSE for signal quality
            
            # Balanced combination - prioritize detection accuracy first
            total_loss = det_loss + 0.01 * enc_loss  # Very small encoder penalty
            return total_loss
        
        # Simpler approach: just train detector loss first
        phase1_model.compile(
            optimizer=tf.keras.optimizers.Nadam(1e-4),
            loss={'phase1_model_1': 'mse', 'phase1_model_2': 'binary_crossentropy'},  # [signal, detection]
            loss_weights={'phase1_model_1': 0.01, 'phase1_model_2': 1.0},  # Prioritize detection
            metrics={'phase1_model_2': [binary_accuracy_metric]}
        )
        
        # Get dataset
        dataset = self.get_improved_dataset()
        
        # Modify dataset for phase 1 (need both signal and message as targets)
        def phase1_targets(inputs, det_target):
            signal, message = inputs
            return inputs, [signal, det_target]  # [signal_target, detection_target]
            
        phase1_ds = dataset.map(phase1_targets)
        
        print(f"Training for {epochs} epochs...")
        history = phase1_model.fit(
            phase1_ds,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=epochs,
            verbose=1
        )
        
        # Check if phase 1 was successful
        final_det_loss = history.history['phase1_model_2_loss'][-1]
        final_det_acc = history.history['phase1_model_2_binary_accuracy_metric'][-1]
        
        print(f"\nPhase 1 Results:")
        print(f"  Detection loss: {final_det_loss:.4f}")
        print(f"  Detection accuracy: {final_det_acc:.4f} ({final_det_acc*100:.1f}%)")
        
        success = final_det_loss < 0.1 and final_det_acc > 0.95
        if success:
            print("✅ Phase 1 SUCCESSFUL: Encoder-detector pair established!")
        else:
            print("❌ Phase 1 FAILED: Need more training or parameter adjustment")
            if final_det_acc > 0.8:
                print("   → Good progress, may just need more epochs")
            
        return success, history
        
    def phase2_gradual_attack_introduction(self, epochs=5):
        """Phase 2: Gradually introduce attacks while training detector"""
        print(f"\n{'='*60}")
        print("PHASE 2: GRADUAL ATTACK INTRODUCTION")
        print(f"{'='*60}")
        print("Gradually introducing attacks while training detector to be robust")
        
        # Create training model with controllable attack strength
        inp_sig = tf.keras.Input((FRAME_LEN,), name='input_signal')
        inp_msg = tf.keras.Input((1, 1, NUM_BITS), name='input_message')
        attack_strength = tf.Variable(0.0, trainable=False, name='attack_strength')
        
        # Encoder output (now frozen to preserve learned embeddings)
        self.encoder.trainable = False
        self.detector.trainable = True
        
        enc_out = self.encoder([inp_sig, inp_msg])
        
        # Gradual attack application
        def gradual_attacks(encoded_signal, strength):
            # Apply attacks with gradually increasing strength
            # Start with no attacks (strength=0) and gradually increase to full strength (strength=1)
            
            # Get clean and fully attacked signals
            clean_signal = encoded_signal
            attacked_signal = self.attacks(encoded_signal, training=True)
            
            # Interpolate between clean and attacked
            mixed_signal = (1.0 - strength) * clean_signal + strength * attacked_signal
            return mixed_signal
            
        # Apply gradual attacks
        mixed_out = tf.py_function(
            lambda x: gradual_attacks(x, attack_strength),
            [enc_out],
            tf.float32
        )
        mixed_out.set_shape(enc_out.shape)
        
        det_out = self.detector(mixed_out)
        
        phase2_model = tf.keras.Model([inp_sig, inp_msg], det_out, name='phase2_model')
        phase2_model.compile(
            optimizer=tf.keras.optimizers.Nadam(2e-4),
            loss='binary_crossentropy',
            metrics=[binary_accuracy_metric]
        )
        
        dataset = self.get_improved_dataset()
        
        print(f"Training for {epochs} epochs with gradual attack introduction...")
        
        epoch_results = []
        for epoch in range(epochs):
            # Gradually increase attack strength
            current_strength = epoch / (epochs - 1) if epochs > 1 else 1.0
            attack_strength.assign(current_strength)
            
            print(f"\nEpoch {epoch+1}/{epochs} - Attack strength: {current_strength:.2f}")
            
            history = phase2_model.fit(
                dataset,
                steps_per_epoch=STEPS_PER_EPOCH // 2,  # Shorter epochs for gradual training
                epochs=1,
                verbose=1
            )
            
            epoch_results.append({
                'epoch': epoch + 1,
                'attack_strength': current_strength,
                'loss': history.history['loss'][0],
                'accuracy': history.history['binary_accuracy_metric'][0]
            })
        
        # Check final performance
        final_loss = epoch_results[-1]['loss']
        final_acc = epoch_results[-1]['accuracy']
        
        print(f"\nPhase 2 Results:")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Final accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
        
        success = final_loss < 0.3 and final_acc > 0.7  # More lenient since attacks are now active
        if success:
            print("✅ Phase 2 SUCCESSFUL: Detector learns robustness to attacks!")
        else:
            print("❌ Phase 2 PARTIAL: Detector struggling with attacks")
            print("   → May need to reduce attack aggressiveness or train longer")
            
        return success, epoch_results
        
    def phase3_end_to_end_training(self, epochs=10):
        """Phase 3: End-to-end training with proper loss weighting"""
        print(f"\n{'='*60}")
        print("PHASE 3: END-TO-END TRAINING WITH BALANCED LOSSES")
        print(f"{'='*60}")
        print("Training encoder and detector together with attacks enabled")
        print("Using dynamic loss weights as specified in the paper")
        
        # Enable training for both networks
        self.encoder.trainable = True
        self.detector.trainable = True
        
        # Create full end-to-end model
        inp_sig = tf.keras.Input((FRAME_LEN,), name='input_signal')
        inp_msg = tf.keras.Input((1, 1, NUM_BITS), name='input_message')
        
        enc_out = self.encoder([inp_sig, inp_msg])
        atk_out = self.attacks(enc_out, training=True)
        det_out = self.detector(atk_out)
        
        final_model = tf.keras.Model([inp_sig, inp_msg], [enc_out, det_out], name='final_model')
        
        # Dynamic loss weights (from debugging analysis)
        def dynamic_loss_weights(epoch):
            # Start with emphasis on detection, gradually balance
            w_d = 3.0  # Detection weight (constant)
            w_e = 0.2 + (epoch / epochs) * 0.8  # Encoder weight (0.2 -> 1.0)
            return w_e, w_d
        
        # Custom training loop for dynamic weights
        optimizer = tf.keras.optimizers.Nadam(1e-4)
        bce_loss = tf.keras.losses.BinaryCrossentropy()
        
        dataset = self.get_improved_dataset()
        
        epoch_results = []
        
        for epoch in range(epochs):
            w_e, w_d = dynamic_loss_weights(epoch)
            print(f"\nEpoch {epoch+1}/{epochs} - Loss weights: w_e={w_e:.2f}, w_d={w_d:.2f}")
            
            epoch_losses = []
            epoch_accs = []
            epoch_snrs = []
            
            for step, (inputs, targets) in enumerate(dataset.take(STEPS_PER_EPOCH)):
                with tf.GradientTape() as tape:
                    signal, message = inputs
                    
                    # Forward pass
                    enc_out = self.encoder([signal, message])
                    atk_out = self.attacks(enc_out, training=True)
                    det_out = self.detector(atk_out)
                    
                    # Losses
                    detection_loss = bce_loss(targets, det_out)
                    encoder_loss = tf.reduce_mean(tf.square(signal - enc_out))  # MSE for signal quality
                    
                    # Combined loss with dynamic weights
                    total_loss = w_d * detection_loss + w_e * encoder_loss
                
                # Gradients and update
                trainable_vars = (self.encoder.trainable_variables + 
                                self.detector.trainable_variables)
                gradients = tape.gradient(total_loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients, trainable_vars))
                
                # Metrics
                acc = binary_accuracy_metric(targets, det_out)
                snr = snr_metric(signal, enc_out)
                
                epoch_losses.append(total_loss.numpy())
                epoch_accs.append(acc.numpy())
                epoch_snrs.append(snr.numpy())
                
                if step % 100 == 0:
                    print(f"  Step {step}/{STEPS_PER_EPOCH} - Loss: {total_loss:.4f}, Acc: {acc:.3f}, SNR: {snr:.1f}dB")
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accs)
            avg_snr = np.mean(epoch_snrs)
            
            epoch_results.append({
                'epoch': epoch + 1,
                'w_e': w_e,
                'w_d': w_d,
                'loss': avg_loss,
                'accuracy': avg_acc,
                'snr_db': avg_snr
            })
            
            print(f"  Epoch Summary - Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}, SNR: {avg_snr:.1f}dB")
        
        # Final evaluation
        final_result = epoch_results[-1]
        print(f"\nPhase 3 Final Results:")
        print(f"  Loss: {final_result['loss']:.4f}")
        print(f"  Accuracy: {final_result['accuracy']:.4f} ({final_result['accuracy']*100:.1f}%)")
        print(f"  SNR: {final_result['snr_db']:.1f} dB")
        
        # Success criteria: good detection accuracy with reasonable signal quality
        success = (final_result['accuracy'] > 0.85 and 
                  final_result['snr_db'] > 15.0 and
                  final_result['loss'] < 1.0)
        
        if success:
            print("✅ Phase 3 SUCCESSFUL: End-to-end training achieved target performance!")
        else:
            print("❌ Phase 3 NEEDS TUNING: Performance below target")
            if final_result['accuracy'] < 0.85:
                print("   → Low accuracy: Consider reducing attack strength or training longer")
            if final_result['snr_db'] < 15.0:
                print("   → Low SNR: Increase encoder loss weight (w_e)")
                
        return success, epoch_results
        
    def save_model_and_results(self, all_results):
        """Save trained model and results"""
        print(f"\nSaving model and training results...")
        
        # Save model weights
        self.model.save_weights('model_b_fixed_weights.h5')
        print("✓ Model weights saved to model_b_fixed_weights.h5")
        
        # Save training history
        import json
        with open('training_history_fixed.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("✓ Training history saved to training_history_fixed.json")
        
        return True

def main():
    """Main training function"""
    print("🚀 Starting Fixed DNN Audio Watermarking Training")
    print("   Addressing all identified issues from debugging analysis")
    
    # Initialize trainer
    trainer = ProgressiveTraining()
    trainer.build_model()
    
    all_results = {}
    
    # Phase 1: Encoder-Detector Initialization
    success1, history1 = trainer.phase1_encoder_detector_training(epochs=3)
    all_results['phase1'] = history1.history if hasattr(history1, 'history') else history1
    
    if not success1:
        print("❌ Phase 1 failed. Cannot proceed to subsequent phases.")
        return False
    
    # Phase 2: Gradual Attack Introduction  
    success2, history2 = trainer.phase2_gradual_attack_introduction(epochs=5)
    all_results['phase2'] = history2
    
    # Phase 3: End-to-End Training (proceed even if Phase 2 partial success)
    success3, history3 = trainer.phase3_end_to_end_training(epochs=10)
    all_results['phase3'] = history3
    
    # Save everything
    trainer.save_model_and_results(all_results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Phase 1 (Initialization): {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"Phase 2 (Attack Introduction): {'✅ SUCCESS' if success2 else '⚠️ PARTIAL'}")
    print(f"Phase 3 (End-to-End): {'✅ SUCCESS' if success3 else '⚠️ NEEDS TUNING'}")
    
    overall_success = success1 and (success2 or success3)
    if overall_success:
        print("\n🎉 OVERALL TRAINING SUCCESSFUL!")
        print("   Your watermarking system is ready for deployment.")
    else:
        print("\n⚠️ TRAINING PARTIALLY SUCCESSFUL")
        print("   System functional but may benefit from parameter tuning.")
        
    return overall_success

if __name__ == "__main__":
    success = main()