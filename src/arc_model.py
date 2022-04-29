import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from .model import CNN_C_trip
from .layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists
)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def Backbone(opt, input_, backbone):
    """Backbone Model"""
    x = CNN_C_trip(opt, input_, backbone)
    return x


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        x = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer_arcface')(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head


def NormHead(opt=None, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(opt.num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head


def ArcFaceModel(opt=None, 
                 channels=1, 
                 name='arcface_model',
                 margin=0.5, 
                 logist_scale=64, 
                 head_type='ArcHead', 
                 training=True):
    """Arc Face Model"""
    x = inputs = Input([opt.input_shape, 1], name='input_signal')

    x = Backbone(opt, x, True)

    embds = OutputLayer(opt.embedding_size, w_decay=5e-4)(x)

    if training:
        assert opt.num_classes is not None
        labels = Input([], name='label')
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=opt.num_classes, 
                             margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        else:
            logist = NormHead(opt=opt)(embds)
        return Model([inputs, labels], logist, name=name)
    else:
        return Model(inputs, embds, name=name)
