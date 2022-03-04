import tensorflow as tf
import numpy as np
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda
from keras.models import Model
from network.nn import TransformerLayer
from keras import regularizers

from .module import Conv1D, ReLU, ResidualConv1DGLU
from .upsample import UpsampleNetwork
# Reference:  https://github.com/kokeshing/WaveNet-tf2

def WaveNet(opt):
    inputs = Input(shape=[400, 1])
    x = Conv1D(32, kernel_size=1, padding='causal')(inputs)

    skips = None
    for _ in range(2):
        for i in range(10):
            x, h = ResidualConv1DGLU(32, 256, kernel_size=3, skip_out_channels=64, dilation_rate=2 ** i)(x)
            if skips is None:
                skips = h
            else:
                skips = skips + h
    x = skips
    x = tf.keras.layers.ReLU()(x)

    x = Conv1D(32, kernel_size=1, padding='causal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                    bias_regularizer=regularizers.l2(1e-4),
                                                    activity_regularizer=regularizers.l2(1e-5))(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv1D(64, kernel_size=1, padding='causal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                    bias_regularizer=regularizers.l2(1e-4),
                                                    activity_regularizer=regularizers.l2(1e-5))(x)

    x = MaxPooling1D(pool_size=4, strides=None)(x)
    x = GlobalAveragePooling1D()(x)
    if opt.multi_head == True:
      x = TransformerLayer(x=x, c=64, num_heads=64//4)
      x = Dense(opt.num_classes, activation='softmax')(x)
      m = Model(inputs, x, name='wavenet_multi_head')
    else:
      x = Dense(opt.num_classes, activation='softmax')(x)
      m = Model(inputs, x, name='wavenet')
    return m
