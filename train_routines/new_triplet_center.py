######################################################
# Original implementation by KinWaiCheuk: https://github.com/KinWaiCheuk/Triplet-net-keras
######################################################

from preprocessing.utils import to_one_hot
import tensorflow as tf
from tensorflow.keras.models import Model
from triplet import generate_triplet, new_triplet_loss
from tensorflow.keras.layers import concatenate, Lambda, Embedding, Input, BatchNormalization
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import os
import argparse
from angular_grad import AngularGrad
from keras.layers import Dense

def train_new_triplet_center(opt, x_train, y_train, x_test, y_test, network, i=100):
    print("\n Training with Triplet Loss....")

    outdir = opt.outdir + "/new_triplet_loss/"
    if i==0:
      epoch = 50 # 30
    else:
      epoch = opt.epoch # 10

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    model_input = Input(shape=(opt.input_shape, 1))
    softmax, pre_logits = network(opt, model_input)
    
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    shared_model = tf.keras.models.Model(inputs=[model_input], outputs=[softmax, pre_logits])
    # shared_model.summary()
   
    X_train, Y_train = generate_triplet(x_train, y_train)  #(anchors, positive, negative)
    
  
    anchor_input   = Input((opt.input_shape, 1,), name='anchor_input')
    positive_input = Input((opt.input_shape, 1,), name='positive_input')
    negative_input = Input((opt.input_shape, 1,), name='negative_input')
    target_input   = Input((1,), name='target_input')

    soft_anchor, pre_logits_anchor = shared_model([anchor_input])
    soft_pos, pre_logits_pos       = shared_model([positive_input])
    soft_neg, pre_logits_neg       = shared_model([negative_input])

    center = Dense(10, activation='relu', use_bias=False)(target_input)
    if opt.activation == 'softmax':
      center = Dense(opt.embedding_size, activation='softmax', use_bias=False)(center)
    elif opt.activation == 'relu':
      center = Dense(opt.embedding_size, activation='relu', use_bias=False)(center)
    elif opt.activation == 'sigmoid':
      center = Dense(opt.embedding_size, activation=tf.keras.activations.sigmoid, use_bias=False)(center)
    elif opt.activation == 'softplus':
      center = Dense(opt.embedding_size, activation=tf.keras.activations.softplus, use_bias=False)(center)
    elif opt.activation == 'softsign':
      center = Dense(opt.embedding_size, activation=tf.keras.activations.softplus, use_bias=False)(center)
    elif opt.activation == 'tanh':
      center = Dense(opt.embedding_size, activation=tf.keras.activations.tanh, use_bias=False)(center)
    elif opt.activation == 'selu':
      center = Dense(opt.embedding_size, activation=tf.keras.activations.selu, use_bias=False)(center)
    elif opt.activation == 'elu':
      center = Dense(opt.embedding_size, activation=tf.keras.activations.elu, use_bias=False)(center)
    elif opt.activation == 'exponential':
      center = Dense(opt.embedding_size, activation=tf.keras.activations.exponential, use_bias=False)(center)
    else:
      center = Dense(opt.embedding_size)(center)
    center = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False)(center)

    center_shared_model = tf.keras.models.Model(inputs=[target_input], outputs=[center])
    center = center_shared_model([target_input])

    merged_pre  = concatenate([pre_logits_anchor, pre_logits_pos, pre_logits_neg, center], axis=-1, name='merged_pre')
    merged_soft = concatenate([soft_anchor, soft_pos, soft_neg], axis=-1, name='merged_soft')
    
    loss_weights = [1, 0.01]

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    model = Model(inputs=[anchor_input, positive_input, negative_input, target_input], outputs=[merged_soft, merged_pre])
    
    # if os.path.isdir(outdir + "new_triplet_loss_model"):
    #   model.load_weights(outdir + "new_triplet_loss_model")
    #   print(f'\n Load weight: {outdir}new_triplet_loss_model')
    # else:
    #   print('\n No weight file.')

    model.compile(loss=["categorical_crossentropy", new_triplet_loss],
                  optimizer=AngularGrad(), metrics=["accuracy"], loss_weights=loss_weights)
    # https://keras.io/api/losses/
    
    # data-----------------------------------------------------
    anchor   = X_train[:, 0, :].reshape(-1, opt.input_shape, 1)
    positive = X_train[:, 1, :].reshape(-1, opt.input_shape, 1)
    negative = X_train[:, 2, :].reshape(-1, opt.input_shape, 1)

    y_anchor   = to_one_hot(Y_train[:, 0])
    y_positive = to_one_hot(Y_train[:, 1])
    y_negative = to_one_hot(Y_train[:, 2])
    y_target   = Y_train[:, 1]


    target = np.concatenate((y_anchor, y_positive, y_negative), -1)

    # Fit data-------------------------------------------------
    model.fit(x=[anchor, positive, negative, y_target], y=[target, y_target],
              batch_size=opt.batch_size, epochs=epoch, callbacks=[TensorBoard(log_dir=outdir)], shuffle=True)
    tf.saved_model.save(model, outdir + 'new_triplet_loss_model')

    # Embedding------------------------------------------------
    model = Model(inputs=[anchor_input], outputs=[soft_anchor, pre_logits_anchor])
    model.load_weights(outdir + "new_triplet_loss_model")

    x_train, y_train = choosing_features(x_train, y_train)
    
    _, X_train_embed = model.predict([x_train])
    y_test_soft, X_test_embed = model.predict([x_test])
    
    from TSNE_plot import tsne_plot
    tsne_plot(outdir, opt.activation, X_train_embed, X_test_embed, y_train, y_test)
    
    return X_train_embed, X_test_embed, y_test_soft
