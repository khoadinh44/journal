from preprocessing.utils import to_one_hot, choosing_features
import tensorflow as tf
from tensorflow.keras.models import Model
from triplet import generate_triplet, triplet_center_loss
from tensorflow.keras.layers import concatenate, Lambda, Embedding, Input
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from angular_grad import AngularGrad
from keras.layers import Dense
import os
import argparse

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)

def train_triplet_center_loss(opt, x_train, y_train, x_test, y_test, network):
    print("\n Training with Triplet Center Loss....")

    outdir = opt.outdir + "/triplet_center_loss/"

    x_input = Input(shape=(opt.input_shape, 1), name='x_input')
    y_train_onehot = to_one_hot(y_train)
    y_train = y_train.astype(np.float32)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    loss_weights = [1, 0.01]

    softmax, pre_logits = network(opt, x_input)
    shared_model = tf.keras.models.Model(inputs=[x_input], outputs=[softmax, pre_logits])
    softmax, pre_logits = shared_model([x_input])

    target_input = Input((1,), name='target_input')
    center = Dense(opt.embedding_size)(target_input)
    shared_model = tf.keras.models.Model(inputs=[target_input], outputs=[center])
    center = shared_model([target_input])

    merged_pre = concatenate([pre_logits, center], axis=-1, name='merged_pre')

    model = tf.keras.models.Model(inputs=[x_input, target_input], outputs=[softmax, merged_pre])

    model.compile(loss=["categorical_crossentropy", triplet_center_loss],
                  optimizer=AngularGrad(), metrics=["accuracy"],
                  loss_weights=loss_weights)
    
    if opt.use_weight:
      if os.path.isdir(outdir + "triplet_center_loss_model"):
          model.load_weights(outdir + "triplet_center_loss_model")
          print(f'\n Load weight: {outdir}')
      else:
          print('\n No weight file.')
    model.fit(x=[x_train, y_train], y=[y_train_onehot, y_train],
              batch_size=opt.batch_size,  
              # allbacks = [callback], 
              epochs=opt.epoch)

    tf.saved_model.save(model, outdir + 'triplet_center_loss_model')
    

    # model = Model(inputs=[x_input], outputs=[softmax, pre_logits])
    # x_train, y_train = choosing_features(x_train, y_train)
    _,           X_train_embed  = model.predict([x_train, y_train])
    y_test_soft, X_test_embed   = model.predict([x_test, y_test])
    
    from TSNE_plot import tsne_plot
    tsne_plot(outdir, "triplet_center_loss_model", X_train_embed, X_test_embed, y_train, y_test)
    y_train = y_train.astype(np.int32)
    return X_train_embed, X_test_embed, y_test_soft, y_train, outdir

