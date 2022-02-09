import tensorflow as tf

import numpy as np
import os

from network.wavenet import WaveNet
from network.module import CrossEntropyLoss


@tf.function
def train_step(model, x, mel_sp, y, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_hat = model(x, mel_sp)
        loss = loss_fn(y, y_hat)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_wavelet(opt):
    os.makedirs(hparams.result_dir + "weights/", exist_ok=True)

    summary_writer = tf.summary.create_file_writer(opt.result_dir)

    wavenet = WaveNet(opt.num_mels, opt.upsample_scales)

    loss_fn = CrossEntropyLoss(num_classes=opt.num_classes)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(opt.learning_rate,
                                                                 decay_steps=opt.exponential_decay_steps,
                                                                 decay_rate=opt.exponential_decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         beta_1=opt.beta_1)

    if hparams.load_path is not None:
        wavenet.load_weights(hparams.load_path)
        step = np.load(hparams.result_dir + "weights/step.npy")
        step = step
        print(f"weights load: {hparams.load_path}")
    else:
        step = 0

    for epoch in range(hparams.epoch):
        train_data = get_train_data()
        for x, mel_sp, y in train_data:
            loss = train_step(wavenet, x, mel_sp, y, loss_fn, optimizer)
            with summary_writer.as_default():
                tf.summary.scalar('train/loss', loss, step=step)

            step += 1

        if epoch % hparams.save_interval == 0:
            print(f'Step {step}, Loss: {loss}')
            np.save(opt.result_dir + f"weights/step.npy", np.array(step))
            wavenet.save_weights(opt.result_dir + f"weights/wavenet_{epoch:04}")

    np.save(opt.result_dir + f"weights/step.npy", np.array(step))
    wavenet.save_weights(opt.result_dir + f"weights/wavenet_{epoch:04}")

if __name__ == '__main__':
    train()
