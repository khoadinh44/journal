from preprocessing.utils import to_one_hot, choosing_features
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
from keras.layers import Dense

def l2_loss(y_true, y_pred):
  total_length = y_pred.shape[1]
  pre_logits, center, mean_var = y_pred[:, :int(total_length * 1/3)], y_pred[:, int(total_length * 1/3): int(total_length * 2/3)], y_pred[:, int(total_length * 2/3):]
  
  out_l2_pre      = K.sum(K.square(pre_logits - center))
  out_l2_mean_var = K.sum(K.square(mean_var - center))
  return out_l2_pre + out_l2_mean_var


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)

def train_new_center_loss(opt, x_train, y_train, x_test, y_test, network):
    print("\n Training with Center Loss....")

    outdir = opt.outdir + "/new_center_loss_model/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    loss_weights = [1, 0.01]
    
    y_train_onehot = to_one_hot(y_train)
    y_train = y_train.astype(np.float32)
    x_train_mean = np.expand_dims(np.mean(x_train, axis=0), axis=-1)
    x_train_var = np.expand_dims(np.var(x_train, axis=0), axis=-1)
    x_train_mean_var = np.concatenate((x_train_mean, x_train_var), axis=-1)
    
    x_test_mean = np.expand_dims(np.mean(x_test, axis=0), axis=-1)
    x_test_var = np.expand_dims(np.var(x_test, axis=0), axis=-1)
    x_test_mean_var = np.concatenate((x_test_mean, x_test_var), axis=-1)
    
    # Input layers------------------------------------------------
    x_input        = Input(shape=(opt.input_shape, 1), name='x_input')
    target_input   = Input((1,), name='target_input')
    mean_var_input = Input((2, ), name='mean_and_variance_input')
    
    
    # Extra Model ----------------------------------------------------
    softmax, pre_logits = network(opt, x_input)
    shared_model = tf.keras.models.Model(inputs=[x_input], outputs=[softmax, pre_logits])
    softmax, pre_logits = shared_model([x_input])

    
    center = Dense(opt.embedding_size)(target_input)
    center_shared_model = tf.keras.models.Model(inputs=[target_input], outputs=[center])
    y_center = center_shared_model([target_input])

    
#     mean_var = Dense(opt.embedding_size//3)(mean_var_input)
    mean_var = Dense(opt.embedding_size)(mean_var_input)
    mean_var_shared_model = tf.keras.models.Model(inputs=[mean_var_input], outputs=[mean_var])
    y_mean_var = mean_var_shared_model([mean_var_input])

    merged_pre = concatenate([pre_logits, y_center, y_mean_var], axis=-1, name='merged_pre')
    merged_pre_mean_var = concatenate([pre_logits, y_mean_var], axis=-1, name='merged_pre_mean_var')

    model = tf.keras.models.Model(inputs=[x_input, target_input, mean_var_input], outputs=[softmax, merged_pre])

    model.compile(loss=["categorical_crossentropy", l2_loss],
                  optimizer=AngularGrad(), 
                  metrics=["accuracy"],
                  loss_weights=loss_weights)
    
    if opt.use_weight:
      if os.path.isdir(outdir + "new_center_loss_model"):
        model.load_weights(outdir + "new_center_loss_model")
        print(f'\n Load weight: {outdir}')
      else:
        print('\n No weight file.')
    
    model.fit(x=[x_train, y_train], y=[y_train_onehot, y_train],
              batch_size=opt.batch_size,  
              # callbacks=[callback],
              epochs=opt.epoch,)

    tf.saved_model.save(model, outdir + 'new_center_loss_model')

    model = Model(inputs=[x_input, mean_var_input], outputs=[softmax, merged_pre_mean_var])
    model.load_weights(outdir + "center_loss_model")

    # x_train, y_train = choosing_features(x_train, y_train)
    _,           X_train_embed  = model.predict([x_train, x_train_mean_var])
    y_test_soft, X_test_embed   = model.predict([x_test, x_test_mean_var])
    
    from TSNE_plot import tsne_plot
    tsne_plot(outdir, "new_center_loss_model", X_train_embed, X_test_embed, y_train, y_test)

    y_train = y_train.astype(np.int32)
    return X_train_embed, X_test_embed, y_test_soft, y_train, outdir
