##########################################################################
### Original implementation by shamangary: https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization
##########################################################################

from preprocessing.utils import to_one_hot
import tensorflow as tf
from tensorflow.keras.models import Model
from triplet import generate_triplet, triplet_center_loss
from tensorflow.keras.layers import concatenate, Lambda, Embedding, Input
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from angular_grad import AngularGrad
import os
import argparse


def train_center_loss(opt, x_train, y_train, x_test, y_test, network):
    print("\n Training with Center Loss....")

    outdir = opt.outdir + "/center_loss/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    loss_weights = [1, 0.1]

    x_input = Input(shape=(opt.input_shape, 1))
    y_train_onehot = to_one_hot(y_train)
    
    softmax, pre_logits = network(opt, x_input)
    target_input = Input((1,), name='target_input')
    center = Embedding(10, opt.embedding_size)(target_input)
    
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([pre_logits, center])
    
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    model = tf.keras.models.Model(inputs=[x_input, target_input], outputs=[softmax, l2_loss])
    model.compile(loss=["categorical_crossentropy", lambda y_true, y_pred: y_pred],
                  optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"],
                  loss_weights=loss_weights)

    model.fit([x_train, y_train], y=[y_train_onehot, y_train],
              batch_size=opt.batch_size, epochs=opt.epoch, callbacks=[TensorBoard(log_dir=outdir)], validation_split=0.2)

    model.save(outdir + "center_loss_model.h5")

    model = Model(inputs=[x_input, target_input], outputs=[softmax, l2_loss, pre_logits])
    model.load_weights(outdir + "center_loss_model.h5")

    _, _, X_train_embed          = model.predict([x_train, y_train])
    y_test_soft, _, X_test_embed = model.predict([x_test, y_test])

    return X_train_embed, X_test_embed, y_test_soft
