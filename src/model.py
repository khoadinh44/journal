import os
from functools import partial
import tensorflow as tf
import keras
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate
import keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# def TransformerLayer(x=None, c=48, num_heads=4*3, backbone=None):
#     # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)

#     q   = Dense(c, use_bias=True, 
#                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                   bias_regularizer=regularizers.l2(1e-4),
#                   activity_regularizer=regularizers.l2(1e-5))(x)
#     k   = Dense(c, use_bias=True, 
#                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                   bias_regularizer=regularizers.l2(1e-4),
#                   activity_regularizer=regularizers.l2(1e-5))(x)
#     v   = Dense(c, use_bias=True, 
#                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                   bias_regularizer=regularizers.l2(1e-4),
#                   activity_regularizer=regularizers.l2(1e-5))(x)
#     ma  = MultiHeadAttention(head_size=c, num_heads=num_heads)([q, k, v]) 
#     fc1 = Dense(c, use_bias=True, 
#                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                   bias_regularizer=regularizers.l2(1e-4),
#                   activity_regularizer=regularizers.l2(1e-5))(ma)
#     if backbone:
#       fc1 = tf.keras.layers.Dropout(0.5)(fc1)    
#     else:
#       fc1 = tf.keras.layers.Dropout(0.2)(fc1)                         
#     fc2 = Dense(c, use_bias=True, 
#                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                   bias_regularizer=regularizers.l2(1e-4),
#                   activity_regularizer=regularizers.l2(1e-5))(fc1) + x
#     if backbone:
#       fc2 = tf.keras.layers.Dropout(0.5)(fc2)    
#     else:
#       fc2 = tf.keras.layers.Dropout(0.2)(fc2) 
#     return fc2

# # For m34 Residual, use RepeatVector. Or tensorflow backend.repeat
# def identity_block(input_tensor, kernel_size, filters, stage, block):
#     conv_name_base = 'res' + str(stage) + str(block) + '_branch'
#     bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

#     x = Conv1D(filters,
#                kernel_size=kernel_size,
#                strides=1,
#                padding='same',
#               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#               bias_regularizer=regularizers.l2(1e-4),
#               activity_regularizer=regularizers.l2(1e-5),
#               name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)

#     x = Conv1D(filters,
#                kernel_size=kernel_size,
#                strides=1,
#                padding='same',
#               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#               bias_regularizer=regularizers.l2(1e-4),
#               activity_regularizer=regularizers.l2(1e-5),
#               name=conv_name_base + '2b')(x)
#     x = BatchNormalization(name=bn_name_base + '2b')(x)

#     if input_tensor.shape[2] != x.shape[2]:
#         x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
#     else:
#         x = layers.add([x, input_tensor])

#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     # x = tf.keras.activations.gelu(x)
#     return x

# def CNN_C_trip(opt, input_, backbone=False):
#     '''
#     The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
#     https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
#     '''
#     x = Conv1D(48,
#                kernel_size=80,
#                strides=4,
#                padding='same',
#                kernel_initializer='glorot_uniform',
#                kernel_regularizer=regularizers.l2(l=0.0001),)(input_)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = MaxPooling1D(pool_size=4, strides=None)(x)

#     for i in range(3):
#         x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i)


#     x = MaxPooling1D(pool_size=4, strides=None)(x)
#     x = GlobalAveragePooling1D()(x)
#     x = TransformerLayer(x=x, c=48, backbone=backbone)
#     if backbone:
#         return x
#     x = Dense(opt.embedding_size)(x)
#     x = BatchNormalization()(x)
#     pre_logit = x
#     softmax = Dense(opt.num_classes, activation='softmax')(x)

#     return softmax, pre_logit



def TransformerLayer(x=None, c=48, num_heads=4, backbone=None):
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
    ma  = MultiHeadAttention(head_size=c, num_heads=num_heads)([q, k, v]) 
    fc2 = tf.keras.layers.Dropout(0.5)(ma) 

    return fc2

# For m34 Residual, use RepeatVector. Or tensorflow backend.repeat
def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
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
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)

    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def CNN_C_trip(opt, input_, backbone=False):
    '''
    The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
    https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
    '''
    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),)(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(2):
        x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(2):
        x = identity_block(x, kernel_size=3, filters=96, stage=3, block=i)

    x = MaxPooling1D(pool_size=4, strides=None)(x)
    
    # x = GlobalAveragePooling1D()(x)
    x = GlobalAveragePooling1D(data_format='channels_first', keepdims=False)(x)
    x = TransformerLayer(x=x, c=244, backbone=backbone)
    
    if backbone:
        return x
    
    x = Dense(opt.embedding_size)(x)
    x = BatchNormalization()(x)
    pre_logit = x
    softmax = Dense(opt.num_classes, activation='softmax')(x)

    return softmax, pre_logit
