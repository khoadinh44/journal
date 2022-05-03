from preprocessing.utils import to_one_hot, choosing_features, scaler_transform
from preprocessing.extracted_signal import extracted_feature_of_signal
import tensorflow as tf
from tensorflow.keras.models import Model
from triplet import generate_triplet, triplet_center_loss
from tensorflow.keras.layers import concatenate, Lambda, Embedding, Input
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from angular_grad import AngularGrad
from sklearn.preprocessing import PowerTransformer
import os
import argparse
from keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.layers import concatenate, Lambda, Embedding, Input, BatchNormalization, Dropout, GlobalAveragePooling1D

def l2_loss(y_true, y_pred):
  total_length = y_pred.shape[1]
  pre_logits, center = y_pred[:, :int(total_length/2)], y_pred[:, int(total_length/2): ]
  print(total_length)
  out_l2_pre      = K.sum(K.square(pre_logits - center))
  return out_l2_pre


def extracted_model(in_, opt):
  q   = Dense(opt.embedding_size, use_bias=True, 
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5))(in_)
  k   = Dense(opt.embedding_size, use_bias=True, 
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5))(in_)
  v   = Dense(opt.embedding_size, use_bias=True, 
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5))(in_)
  x  = MultiHeadAttention(head_size=opt.embedding_size, num_heads=12)([q, k, v]) 
  x = BatchNormalization()(x)
  x = Dropout(rate=0.5)(x)
  x = Dense(opt.embedding_size)(x)
  x = BatchNormalization()(x)
  x = Lambda(lambda  x: K.l2_normalize(x, axis=1))(x)
  return x


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)

def train_new_center_loss(opt, x_train_scale, x_train, y_train, x_test_scale, x_test, y_test, network):
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

    x_train_extract = scaler_transform(x_train_extract, PowerTransformer)
    x_test_extract  = scaler_transform(x_test_extract, PowerTransformer)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/x_train_extract.npy', 'wb') as f:
      np.save(f, x_train_extract)
    with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/x_test_extract.npy', 'wb') as f:
      np.save(f, x_test_extract)

    
    print(f'x_train_extract shape: {x_train_extract.shape}')
    print(f'x_test_extract shape: {x_test_extract.shape}')
     
    # Main model ----------------------------------------------------
    x_input = Input(shape=(opt.input_shape, 1), name='x_input')
    softmax, pre_logits = network(opt, x_input)
    shared_model = tf.keras.models.Model(inputs=[x_input], outputs=[softmax, pre_logits])
    shared_model.summary()
    softmax, pre_logits = shared_model([x_input])

    # Target model----------------------------------------------
    center_shared_model = tf.keras.Sequential()
    target_input   = Input((1,), name='target_input')
    # center = Dense(opt.embedding_size)(target_input)
    # center = Lambda(lambda  x: K.l2_normalize(x, axis=1))(center)
    # center_shared_model = tf.keras.models.Model(inputs=[target_input], outputs=[center])
    center_shared_model.add(Input((1,), name='target_input'))
    center_shared_model.add(tf.keras.layers.Embedding(10, opt.embedding_size*2))
    center_shared_model.add(GlobalAveragePooling1D())
    center_shared_model.add(Lambda(lambda  x: K.l2_normalize(x, axis=1)))
    y_center = center_shared_model([target_input])

    # Extract model--------------------------------------------------
    extract_input = Input((11, ), name='extract_input')
    extract_model  = extracted_model(extract_input, opt)
    extract_shared_model = tf.keras.models.Model(inputs=[extract_input], outputs=[extract_model])
    y_extract = extract_shared_model([extract_input])

    merged_pre_extract = concatenate([pre_logits, y_extract], axis=-1)
    # merged_pre_extract = Dense(opt.embedding_size)(merged_pre_extract)
    # merged_pre_extract = Dropout(rate=0.5)(merged_pre_extract)
    # merged_pre_extract = BatchNormalization()(merged_pre_extract)
    # merged_pre_extract = Lambda(lambda  x: K.l2_normalize(x, axis=1))(merged_pre_extract)
    merged_pre_logits = concatenate([merged_pre_extract, y_center], axis=-1, name='merged_pre')

    # train logic------------------------------------------------------------------------------------------------
    model = tf.keras.models.Model(inputs=[x_input, extract_input, target_input], outputs=[softmax, merged_pre_logits])

    model.compile(loss=["categorical_crossentropy", l2_loss],
                  optimizer=tf.keras.optimizers.SGD(), 
                  metrics=["accuracy"],
                  loss_weights=loss_weights)
    
    if opt.use_weight:
      if os.path.isdir(outdir + "new_center_loss"):
        model.load_weights(outdir + "new_center_loss")
        print(f'\n Load weight: {outdir}/new_center_loss')
      else:
        print('\n No weight file.')
    
    model.fit(x=[x_train_scale, x_train_extract, y_train], y=[y_train_onehot, y_train],
              validation_data=([x_test_scale, x_test_extract, y_test], [y_test_onehot, y_test]),
              batch_size=opt.batch_size,  
              # callbacks=[callback],
              epochs=opt.epoch,)

    tf.saved_model.save(model, outdir + 'new_center_loss')

    # from input data---------------------------
    model = Model(inputs=[x_input, extract_input], outputs=[softmax, merged_pre_extract])
    model.load_weights(outdir + "new_center_loss")

    _,           X_train_embed = model.predict([x_train_scale, x_train_extract])
    y_test_soft, X_test_embed = model.predict([x_test_scale, x_test_extract])

    
    from TSNE_plot import tsne_plot
    tsne_plot(outdir, "new_center_loss", X_train_embed, X_test_embed, y_train, y_test)

    y_train = y_train.astype(np.int32)
    return X_train_embed, X_test_embed, y_test_soft, y_train, outdir
