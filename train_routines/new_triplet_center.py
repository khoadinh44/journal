######################################################
# Original implementation by KinWaiCheuk: https://github.com/KinWaiCheuk/Triplet-net-keras
######################################################


import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os
import argparse
from sklearn.preprocessing import PowerTransformer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Lambda, Embedding, Input, BatchNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from keras.layers import Dense

from triplet import generate_triplet, new_triplet_loss
from preprocessing.extracted_signal import extracted_feature_of_signal
from preprocessing.utils import to_one_hot, choosing_features, scaler_transform
from angular_grad import AngularGrad


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)

def extracted_model(in_, opt):
  x = Dense(opt.embedding_size*2,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5))(in_)
  x = Dense(opt.embedding_size*4,
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
  x = concatenate([x, in_], axis=-1)
  x = Dropout(rate=0.5)(x)
  x = Dense(opt.embedding_size)(x)
  x = BatchNormalization()(x)
  # x = Lambda(lambda  x: K.l2_normalize(x, axis=1))(x)
  return x

def train_new_triplet_center(opt, x_train_scale, x_train, y_train, x_test_scale, x_test, y_test, network, i=100):
    print("\n Training with Triplet Loss....")

    outdir = opt.outdir + "/new_triplet_loss/"
    if i==0:
      epoch = 50 # 30
    else:
      epoch = opt.epoch # 10

    if not os.path.isdir(outdir):
        os.makedirs(outdir)


    # Get extracted data for testing-------------------------------
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
    
    # Extract model ---------------------------------------------------------
    extract_input_1 = Input((11), name='extract_input_1')
    extract_model_1  = extracted_model(extract_input_1, opt)
    extract_shared_model_1 = tf.keras.models.Model(inputs=[extract_input_1], outputs=[extract_model_1])
    y_extract_1 = extract_shared_model_1([extract_input_1])

    extract_input_2 = Input((11), name='extract_input_2')
    extract_model_2  = extracted_model(extract_input_2, opt)
    extract_shared_model_2 = tf.keras.models.Model(inputs=[extract_input_2], outputs=[extract_model_2])
    y_extract_2 = extract_shared_model_2([extract_input_2])


    # Center model----------------------------------------------------------
    target_input = Input((1,), name='target_input')
    center_shared_model = tf.keras.Sequential()
    center_shared_model.add(target_input)
    center_shared_model.add(tf.keras.layers.Embedding(1000, opt.embedding_size))
    center_shared_model.add(GlobalAveragePooling1D())
    # center_shared_model.add(Lambda(lambda  x: K.l2_normalize(x, axis=1)))
    center = center_shared_model([target_input])


    # Triplet model----------------------------------------------------------
    model_input = Input(shape=(opt.input_shape, 1))
    softmax, pre_logits = network(opt, model_input)
    shared_model = tf.keras.models.Model(inputs=[model_input], outputs=[softmax, pre_logits])
    shared_model.summary()
    
  
    anchor_input   = Input((opt.input_shape, 1,), name='anchor_input')
    negative_input = Input((opt.input_shape, 1,), name='negative_input')
    
    soft_anchor, pre_logits_anchor = shared_model([anchor_input])
    soft_neg, pre_logits_neg       = shared_model([negative_input])

    merged_pre  = concatenate([pre_logits_anchor, pre_logits_neg, center], axis=-1, name='merged_pre')
    merged_soft = concatenate([soft_anchor, soft_neg], axis=-1, name='merged_soft')
    
    loss_weights = [1, 0.01]
  
    # https://keras.io/api/losses/
    
    # Data for triplet loss----------------------------------------------------

    if os.path.exists('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/anchor_extract.npy'):
      anchor_extract = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/anchor_extract.npy')
      anchor = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/anchor.npy')
      Y_train = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Y_train.npy')
    else:
      # Get generator------------------------------------------------
      X_train, Y_train = generate_triplet(x_train, y_train)  #(anchors, positive, negative)
      anchor   = X_train[:, :X_train.shape[1]//2]
      negative = X_train[:, X_train.shape[1]//2: ]

      anchor_extract = scaler_transform(extracted_feature_of_signal(np.squeeze(anchor)), PowerTransformer)
      anchor_extract = np.squeeze(anchor_extract)
      anchor         = scaler_transform(anchor, PowerTransformer)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/anchor_extract.npy', 'wb') as f:
        np.save(f, anchor_extract)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/anchor.npy', 'wb') as f:
        np.save(f, anchor)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/Y_train.npy', 'wb') as f:
        np.save(f, Y_train)

    print(f'anchor shape: {anchor.shape}')
    print(f'anchor-extract shape: {anchor_extract.shape}\n')
    

    if os.path.exists('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/negative_extract.npy'):
      negative_extract = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/negative_extract.npy')
      negative = np.load('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/negative.npy')
    else:
      negative_extract = scaler_transform(extracted_feature_of_signal(np.squeeze(negative)), PowerTransformer)
      negative_extract = np.squeeze(negative_extract)
      negative         = scaler_transform(negative, PowerTransformer)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/negative_extract.npy', 'wb') as f:
        np.save(f, negative_extract)
      with open('/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/negative.npy', 'wb') as f:
        np.save(f, negative)

    print(f'negative shape: {negative.shape}')
    print(f'negative-extract shape: {negative_extract.shape}\n')

    y_anchor   = to_one_hot(Y_train[:, 0])
    y_negative = to_one_hot(Y_train[:, 1])
    y_target   = Y_train[:, 0]

    target = np.concatenate((y_anchor, y_negative), -1)

    # Fit data-------------------------------------------------
    model = Model(inputs=[anchor_input, negative_input, target_input], outputs=[merged_soft, merged_pre])
    if opt.use_weight:
      if os.path.isdir(outdir + "new_triplet_loss_model"):
        model.load_weights(outdir + "new_triplet_loss_model")
        print(f'\n Load weight : {outdir}')
      else:
        print('\n No weight file.')

    model.compile(loss=["categorical_crossentropy",
                  new_triplet_loss],
                  optimizer=tf.keras.optimizers.SGD(momentum=0.9), 
                  metrics=["accuracy"], 
                  loss_weights=loss_weights)

    model.fit(x=[anchor, anchor_extract, negative, negative_extract, y_target], y=[target, y_target],
              batch_size=opt.batch_size, epochs=epoch, 
              # callbacks=[callback], 
              shuffle=True)
    tf.saved_model.save(model, outdir + 'new_triplet_loss_model')


    # Embedding------------------------------------------------
#     pre_anchor = concatenate([pre_logits_anchor, y_extract_1], axis=-1, name='merged_soft')
    model = Model(inputs=[anchor_input], outputs=[soft_anchor, pre_logits_anchor])
    model.load_weights(outdir + "new_triplet_loss_model")

    # x_train, y_train = choosing_features(x_train, y_train)
    
    _, X_train_embed = model.predict([x_train_scale])
    y_test_soft, X_test_embed = model.predict([x_test_scale])
    
    from TSNE_plot import tsne_plot
    tsne_plot(outdir, 'original', X_train_embed, X_test_embed, y_train, y_test)
#     tsne_plot(outdir, 'extracted', X_train_embed[:, opt.embedding_size: ], X_test_embed[:, opt.embedding_size: ], y_train, y_test)
    
    return X_train_embed, X_test_embed, y_test_soft, y_train, outdir
