from tensorflow import keras

def network():
  # Build neural network
  input_ = keras.layers.Input(shape=[12,])
  hidden1 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU(), kernel_regularizer='l1_l2', kernel_initializer="he_normal")(input_) # "lecun_normal"
  hidden1 = keras.layers.AlphaDropout(rate=0.2)(hidden1),
  # norm1 = keras.layers.BatchNormalization()(hidden1)
  hidden2 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU(), kernel_regularizer='l1_l2', kernel_initializer="he_normal")(hidden1)
  hidden2 = keras.layers.AlphaDropout(rate=0.2)(hidden2),
  # norm2 = keras.layers.BatchNormalization()(hidden2)
  concat = keras.layers.concatenate([input_, hidden2])
  output = keras.layers.Dense(6, activation=tf.keras.layers.ReLU())(concat)
  model = keras.models.Model(inputs=[input_], outputs=[output])
  return model
