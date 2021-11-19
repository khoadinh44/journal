import keras
import tensorflow as tf

'''
After 200 epochs merge_network() was more overstanding than network(), 
with reach up more than 93% validation accuracy while the accuracy of network() was just ~40%.

network+None | merge_network+None | merge_network+Fourie 
40%           93%                   99%
'''

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

def merge_network():
  input_A = keras.layers.Input(shape=[623], name="wide_input")
  input_B = keras.layers.Input(shape=[623], name="deep_input")
  hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
  hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
  concat = keras.layers.concatenate([input_A, hidden2])
  output = keras.layers.Dense(6, name="output")(concat)
  model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
  return model
