import tensorflow as tf
from keras import regularizers

class UpsampleCond(tf.keras.Model):
    def __init__(self, rate, **kwargs):
        super().__init__()

        self.upsampling = tf.keras.layers.UpSampling2D((1, rate), interpolation='nearest')

        self.conv = tf.keras.layers.Conv2D(1, kernel_size=(1, rate * 2 + 1),
                                      padding='same', use_bias=False,
                                      kernel_initializer=tf.constant_initializer(1. / (rate * 2 + 1)),
                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                      bias_regularizer=regularizers.l2(1e-4),
                                      activity_regularizer=regularizers.l2(1e-5))

    @tf.function
    def call(self, x):
        return self.conv(self.upsampling(x))


class UpsampleNetwork(tf.keras.Model):
    def __init__(self, upsample_scales, **kwargs):
        super().__init__()

        self.upsample_layers = [UpsampleCond(scale) for scale in upsample_scales]

    @tf.function
    def call(self, feat):
        for layer in self.upsample_layers:
            feat = layer(feat)

        return feat
