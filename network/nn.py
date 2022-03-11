from functools import partial
import keras
import tensorflow as tf
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate
import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda
from keras.models import Model

def TransformerLayer(x=None, c=48, num_heads=4*3):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    q   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    k   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    v   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    ma  = MultiHeadAttention(head_size=c, num_heads=num_heads)([q, k, v]) + x
    fc1 = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(ma) 
#     fc1 = tf.keras.layers.Dropout(0.5)(fc1)                           
    fc2 = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(fc1) + x
#     fc2 = tf.keras.layers.Dropout(0.5)(fc2) 
    return fc2

# For m34 Residual, use RepeatVector. Or tensorflow backend.repeat
def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              #  kernel_initializer='glorot_uniform',
              #  kernel_regularizer=regularizers.l2(l=0.0001),
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              #  kernel_initializer='glorot_uniform',
              #  kernel_regularizer=regularizers.l2(l=0.0001),
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)

    # up-sample from the activation maps.
    # otherwise it's a mismatch. Recommendation of the authors.
    # here we x2 the number of filters.
    # See that as duplicating everything and concatenate them.
    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def CNN_A(opt):
    '''
    The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
    https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
    '''
    inputs = Input(shape=[opt.input_shape, 1])
    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
              #  kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(opt.num_classes, activation='softmax')(x)

    m = Model(inputs, x, name='resnet34')
    return m

def CNN_B(num_classes, opt):
  DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")
  model = keras.models.Sequential([
            DefaultConv2D(filters=256, kernel_size=7, input_shape=[128, 128, 1]), #
            tf.keras.layers.ReLU(), #
            keras.layers.MaxPooling2D(pool_size=3), #
            keras.layers.Dropout(0.5), #
            DefaultConv2D(filters=256), #
            tf.keras.layers.ReLU(), #
            keras.layers.MaxPooling2D(pool_size=2),#
            keras.layers.Flatten(),
            keras.layers.Dense(units=512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=num_classes, activation='softmax'),])
  return model

def DNN_A(opt):
  input_ = keras.layers.Input(shape=[opt.input_shape, ])
  hidden1 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(input_)
  hidden2 = keras.layers.Dense(100,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(hidden1)
  concat = keras.layers.concatenate([input_, hidden2])
  output = keras.layers.Dense(opt.num_classes, activation=tf.keras.layers.Softmax())(concat)
  model = keras.models.Model(inputs=[input_], outputs=[output])
  return model

def DNN_B(opt):
  input_A = keras.layers.Input(shape=[200], name="wide_input")
  input_B = keras.layers.Input(shape=[200], name="deep_input") 
  hidden1 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU())(input_B)
  hidden2 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU())(hidden1)
  concat = keras.layers.concatenate([input_A, hidden2])
  output = keras.layers.Dense(opt.num_classes, activation=tf.keras.layers.Softmax(), name="output")(concat)
  model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
  return model

def CNN_C(opt):
    '''
    The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
    https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
    '''
    inputs = Input(shape=[opt.input_shape, 1])
    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)
    x = GlobalAveragePooling1D()(x)
    
    x = TransformerLayer(x, c=48)
    x = Dense(opt.num_classes, activation='softmax')(x)

    m = Model(inputs, x, name='resnet34')
    return m

# def CNN_C(opt):
#     '''
#     The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
#     https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
#     '''
#     inputs = Input(shape=[opt.input_shape, 1])
#     x = Conv1D(48,
#                kernel_size=80,
#                strides=4,
#                padding='same',
#                kernel_initializer='glorot_uniform',
#                kernel_regularizer=regularizers.l2(l=0.0001),)(inputs)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = MaxPooling1D(pool_size=4, strides=None)(x)

#     for i in range(3):
#         x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i)

#     x = MaxPooling1D(pool_size=4, strides=None)(x)

#     for i in range(4):
#         x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i)

#     x = MaxPooling1D(pool_size=4, strides=None)(x)

#     for i in range(6):
#         x = identity_block(x, kernel_size=3, filters=192, stage=3, block=i)

#     x = MaxPooling1D(pool_size=4, strides=None)(x)

#     for i in range(3):
#         x = identity_block(x, kernel_size=3, filters=384, stage=4, block=i)

#     x = GlobalAveragePooling1D()(x)
    
#     x = TransformerLayer(x, c=384)
#     x = Dense(opt.num_classes, activation='softmax')(x)

#     m = Model(inputs, x, name='resnet34')
#     return m
