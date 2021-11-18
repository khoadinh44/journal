import keras
import tensorflow as tf

def network():
  # Build neural network
  input_ = keras.layers.Input(shape=[1246, ])
  hidden1 = keras.layers.Dense(500, activation=tf.keras.layers.ReLU(), kernel_regularizer='l1_l2', kernel_initializer="lecun_normal")(input_) # "lecun_normal"
  norm1 = keras.layers.BatchNormalization()(hidden1)
  hidden2 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU(), kernel_regularizer='l1_l2', kernel_initializer="lecun_normal")(norm1) # "lecun_normal"
  norm2 = keras.layers.BatchNormalization()(hidden2)
  hidden3 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU(), kernel_regularizer='l1_l2', kernel_initializer="lecun_normal")(norm2)
  norm3 = keras.layers.BatchNormalization()(hidden3)
  hidden4 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU(), kernel_regularizer='l1_l2', kernel_initializer="lecun_normal")(norm3)
  norm4 = keras.layers.BatchNormalization()(hidden4)
  concat = keras.layers.concatenate([input_, norm4])
  output = keras.layers.Dense(6, activation=tf.keras.layers.Softmax())(concat)
  model = keras.models.Model(inputs=[input_], outputs=[output])
  return model
