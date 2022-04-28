from preprocessing.utils import to_one_hot, choosing_features
from preprocessing.extracted_signal import extracted_feature_of_signal
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
    print("\n Training with New Center Loss....")

    outdir = opt.outdir + "/new_center_loss/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    loss_weights = [1, 0.01]
    
    y_train_onehot = to_one_hot(y_train)
    y_train = y_train.astype(np.float32)

    y_test_onehot = to_one_hot(y_test)
    y_test = y_test.astype(np.float32)

    if os.path.exists('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/x_test_extract.npy'):
      x_train_extract = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/x_train_extract.npy')
      x_test_extract = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/x_test_extract.npy')
    else:
      x_train_get = np.squeeze(x_train)
      x_train_extract = extracted_feature_of_signal(x_train_get)
      x_test_get = np.squeeze(x_test)
      x_test_extract = extracted_feature_of_signal(x_test_get)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/x_train_extract.npy', 'wb') as f:
        np.save(f, x_train_extract)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/x_test_extract.npy', 'wb') as f:
        np.save(f, x_test_extract)

    print(f'x_train_extract shape: {x_train_extract.shape}')
    print(f'x_test_extract shape: {x_test_extract.shape}')
    
    # Input layers------------------------------------------------
    x_input        = Input(shape=(opt.input_shape, 1), name='x_input')
    target_input   = Input((1,), name='target_input')
    extract_input = Input((11, ), name='extract_input')
    
    
    # Extra Model ----------------------------------------------------
    softmax, pre_logits = network(opt, x_input)
    shared_model = tf.keras.models.Model(inputs=[x_input], outputs=[softmax, pre_logits])
    softmax, pre_logits = shared_model([x_input])

    
    center = Dense(opt.embedding_size)(target_input)
    center_shared_model = tf.keras.models.Model(inputs=[target_input], outputs=[center])
    y_center = center_shared_model([target_input])

    
    extract = Dense(opt.embedding_size//3)(extract_input)
    extract = Dense(opt.embedding_size)(extract)
    extract_shared_model = tf.keras.models.Model(inputs=[extract_input], outputs=[extract])
    y_extract = extract_shared_model([extract_input])

    merged_pre = concatenate([pre_logits, y_center, y_extract], axis=-1, name='merged_pre')

    model = tf.keras.models.Model(inputs=[x_input, extract_input, target_input], outputs=[softmax, merged_pre])

    model.compile(loss=["categorical_crossentropy", l2_loss],
                  optimizer=AngularGrad(), 
                  metrics=["accuracy"],
                  loss_weights=loss_weights)
    
    if opt.use_weight:
      if os.path.isdir(outdir + "new_center_loss"):
        model.load_weights(outdir + "new_center_loss")
        print(f'\n Load weight: {outdir}/new_center_loss')
      else:
        print('\n No weight file.')
    
    model.fit(x=[x_train, x_train_extract, y_train], y=[y_train_onehot, y_train],
              validation_data=([x_test, x_test_extract, y_test], [y_test_onehot, y_test]),
              batch_size=opt.batch_size,  
              # callbacks=[callback],
              epochs=opt.epoch,)

    tf.saved_model.save(model, outdir + 'new_center_loss')

    # from input data---------------------------
    model = Model(inputs=[x_input], outputs=[softmax, pre_logits])
    model.load_weights(outdir + "new_center_loss_model")

    _,           X_train_embed_or  = model.predict([x_train])
    y_test_soft, X_test_embed_or   = model.predict([x_test])

    # from extract features of data ----------------------
    extract_shared_model.load_weights(outdir + "new_center_loss")

    X_train_embed_extract  = extract_shared_model.predict([x_train_extract])
    X_test_embed_extract   = extract_shared_model.predict([x_test_extract])

    X_train_embed = np.concatenate((X_train_embed_or, X_train_embed_extract), axis=-1)
    X_test_embed  = np.concatenate((X_test_embed_or, X_test_embed_extract), axis=-1)
    
    from TSNE_plot import tsne_plot
    tsne_plot(outdir, "new_center_loss", X_train_embed, X_test_embed, y_train, y_test)

    y_train = y_train.astype(np.int32)
    return X_train_embed, X_test_embed, y_test_soft, y_train, outdir
