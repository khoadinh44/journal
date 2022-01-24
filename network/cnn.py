from functools import partial
import keras
import tensorflow as tf

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
  model = keras.models.Sequential()
  model.add(Conv1D(64, 50, strides=1, padding='same', use_bias=True, input_shape=(400, )))
  model.add(Activation('relu'))
  model.add(Conv1D(64, 50, strides=1, padding='same', use_bias=True))
  model.add(Activation('relu'))
  model.add(Conv1D(64, 50, strides=1, padding='same', use_bias=True))
  model.add(Activation('relu'))
  model.add(Conv1D(64, 50, strides=1, padding='same', use_bias=True))
  model.add(Activation('relu'))
  model.add(Conv1D(64, 50, strides=1, padding='same', use_bias=True))
  model.add(Activation('relu'))
  model.add(Dense(50))
  model.add(Activation('relu'))
  model.add(Dense(50))
  model.add(Activation('relu'))
  model.add(Dense(6, activation=tf.keras.layers.Softmax()))
  y = model(X)
  return y

  model.summary()
  return model
