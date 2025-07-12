# model_b/train_b.py
import tensorflow as tf, time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from models_b import build_full
from dataset_b import tf_dataset
from config_b import EPOCHS, STEPS_PER_EPOCH, LR, NUM_BITS, BATCH_SIZE
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError

net = build_full()
opt = tf.keras.optimizers.Nadam(LR)

bce = BinaryCrossentropy()
mae = MeanAbsoluteError()


@tf.function
def step(batch, step_id):
    sig, msg = batch
    with tf.GradientTape() as tape:
        enc, _, dec = net([sig, msg], training=True)
        e_loss = mae(sig, enc)
        d_loss = bce(tf.squeeze(msg, (1, 2)), dec)
        # dynamic weights (paper eqn. (6),(7))
        w_e = tf.cond(step_id < 14_000,
                      lambda: 1.0 + 0.2 * tf.cast(step_id // 1_400, tf.float32),
                      lambda: 2.5)
        w_d = tf.cond(step_id < 14_000,
                      lambda: 3.0 - 0.2 * tf.cast(step_id // 1_400, tf.float32),
                      lambda: 0.5)
        loss = w_e * e_loss + w_d * d_loss
    grads = tape.gradient(loss, net.trainable_variables)
    opt.apply_gradients(zip(grads, net.trainable_variables))
    return loss, e_loss, d_loss


def main():
    print("Starting Model B training...")
    print(f"Configuration: {EPOCHS} epochs, {STEPS_PER_EPOCH} steps/epoch, batch size {BATCH_SIZE}")
    print("L=total loss, Enc=encoder loss, Dec=decoder loss")
    print("-" * 60)
    
    ds = tf_dataset()
    st = time.time()
    step_id = 0
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        ds_iter = iter(ds)  # Create fresh iterator each epoch
        
        for step_in_epoch in range(STEPS_PER_EPOCH):
            loss, e, d = step(next(ds_iter), tf.constant(step_id))
            
            if step_id % 100 == 0:  # Progress updates every 100 steps
                elapsed = time.time() - st
                if step_id > 0:
                    eta = elapsed * (EPOCHS * STEPS_PER_EPOCH - step_id) / step_id
                else:
                    eta = 0
                print(f"  Step {step_id:5d} ({step_in_epoch+1:4d}/{STEPS_PER_EPOCH}) | "
                      f"L={loss:.3f} Enc={e:.3f} Dec={d:.3f} | "
                      f"Time: {elapsed:.0f}s ETA: {eta/60:.1f}m")
            
            step_id += 1
            
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1} completed in {epoch_time:.1f}s, saving checkpoint...")
        net.save(f"model_b/ckpt_epoch_{epoch}")
        
    total_time = time.time() - st
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/3600:.2f}h)")


if __name__ == "__main__":
    main()