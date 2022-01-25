import keras
import tensorflow as tf

def DNN(use_model_A=False, use_model_B=False, use_Wavelet=False):
    if use_Wavelet:
      input_A = keras.layers.Input(shape=[300], name="wide_input")
      input_B = keras.layers.Input(shape=[300], name="deep_input") 
      hidden1 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU())(input_B)
      hidden2 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU())(hidden1)
      concat = keras.layers.concatenate([input_A, hidden2])
      output = keras.layers.Dense(6, activation=tf.keras.layers.Softmax(), name="output")(concat)
      model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
      return model
    elif use_model_A:
      input_ = keras.layers.Input(shape=[400, ])
      hidden3 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU())(input_)
      hidden4 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU())(hidden3)
      concat = keras.layers.concatenate([input_, hidden4])
      output = keras.layers.Dense(6, activation=tf.keras.layers.Softmax())(concat)
      model = keras.models.Model(inputs=[input_], outputs=[output])
      return model
    elif use_model_B:
      input_A = keras.layers.Input(shape=[200], name="wide_input")
      input_B = keras.layers.Input(shape=[200], name="deep_input") 
      hidden1 = keras.layers.Dense(300, activation=tf.keras.layers.ReLU())(input_B)
      hidden2 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU())(hidden1)
      concat = keras.layers.concatenate([input_A, hidden2])
      output = keras.layers.Dense(6, activation=tf.keras.layers.Softmax(), name="output")(concat)
      model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
      return model
