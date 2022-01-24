from functools import partial
import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate

def network():
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
            keras.layers.Dense(units=6, activation='softmax'),])
  
  
def network_1D():
  input_ = keras.layers.Input(shape=[1, 400])
  Conv1D_ = Conv1D(64, 4, strides=1, padding='same', use_bias=True, activation=tf.keras.layers.ReLU())(input_)
  hidden1 = Dense(300, activation=tf.keras.layers.ReLU())(Conv1D_)
  hidden2 = Dense(100, activation=tf.keras.layers.ReLU())(hidden1)
  concat = concatenate([input_, hidden2])
  output = Dense(6, activation=tf.keras.layers.Softmax(), name="output")(concat)
  model = keras.models.Model(inputs=[input_], outputs=[output])
  return model

  model.summary()
  return model
