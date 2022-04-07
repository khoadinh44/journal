from preprocessing.utils import to_one_hot
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.models import Model
from triplet import generate_triplet, triplet_center_loss
from tensorflow.keras.layers import concatenate, Lambda, Embedding
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import os
import argparse


def train_xentropy(opt, x_train, y_train, x_test, y_test, network):
    print("#" * 100)
    print("Training with Categorical CrossEntropy Only Loss....")
    print("#" * 100)

    outdir = outdir + "/xentropy_only_loss/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    x_input = Input(shape=(opt.input_shape, 1))
    y_train_onehot = to_one_hot(y_train)

    softmax, pre_logits = network(opt, model_input)

    model = tf.keras.models.Model(inputs=[x_input], outputs=[softmax])

    model.compile(loss=["categorical_crossentropy"],
                  optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=["accuracy"],)

    model.fit([x_train], y=[y_train_onehot],
              batch_size=opt.batch_size, epochs=opt.epoch, callbacks=[TensorBoard(log_dir=outdir)], validation_split=0.2)

    model.save(outdir + "xentropy_loss_model.h5")

    model = Model(inputs=[x_input], outputs=[softmax, pre_logits])
    model.load_weights(outdir + "xentropy_loss_model.h5")

    _, X_train_embed = model.predict([x_train])
    _, X_test_embed = model.predict([x_test])

    return X_train_embed, X_test_embed
