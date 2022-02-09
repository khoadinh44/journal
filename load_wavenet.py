import tensorflow as tf

import numpy as np
import os

from network.wavenet import WaveNet
from network.module import CrossEntropyLoss
from dataset import get_train_data


@tf.function
def train_step(model, x, y, loss_fn, optimizer, opt):
    with tf.GradientTape() as tape:
        y_hat = model(x, opt.num_mels)
        loss = loss_fn(y, y_hat)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_wavenet(opt, X_train, y_train, X_val, y_val, X_test, y_test)::
    os.makedirs(opt.result_dir + "weights/", exist_ok=True)

    summary_writer = tf.summary.create_file_writer(opt.result_dir)

    wavenet = WaveNet(opt.num_mels, opt.upsample_scales)

    loss_fn = CrossEntropyLoss(num_classes=opt.num_classes)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(opt.learning_rate,
                                                                 decay_steps=opt.exponential_decay_steps,
                                                                 decay_rate=opt.exponential_decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         beta_1=opt.beta_1)

    if opt.load_path is not None:
        wavenet.load_weights(opt.load_path)
        step = np.load(opt.result_dir + "weights/step.npy")
        step = step
        print(f"weights load: {hparams.load_path}")
    else:
        step = 0
        
    steps = X_train.shape[0] // opt.batch_size
    pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
    dataset = get_train_data(X_train, y_train, opt)
    
    for step, inputs in enumerate(dataset):
        x, y = inputs
        if step % steps == 0:
            print(f'Epoch {step // steps + 1}/{opt.epochs}')
            pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
            
        loss = train_step(wavenet, x, y, loss_fn, optimizer, opt)
        pb.add(1, [('loss', loss)])
        
        with summary_writer.as_default():
            tf.summary.scalar('train/loss', loss, step=step)
            
        step += 1
        if step % steps == 0:
            if agg_loss == None:
                agg_loss = loss
                print(" -loss improved from -inf value to {}".format(loss))
                np.save(opt.result_dir + f"weights/step.npy", np.array(step))
                model.save_weights(f"weights/wavenet_{step // steps}.h5")
            else:
                if loss < agg_loss:
                    print(" -loss improved from {} value to {}".format(agg_loss, loss))
                    agg_loss = loss
                    model.save_weights(f"weights/wavenet_{step // steps}.h5")


if __name__ == '__main__':
    train()
