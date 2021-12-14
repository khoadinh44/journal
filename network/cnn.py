from functools import partial
import keras

def network():
  DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")
  model = keras.models.Sequential([
      DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
      keras.layers.MaxPooling2D(pool_size=2),
      DefaultConv2D(filters=128),
      DefaultConv2D(filters=128),
      keras.layers.MaxPooling2D(pool_size=2),
      DefaultConv2D(filters=256),
      DefaultConv2D(filters=256),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Flatten(),
      keras.layers.Dense(units=128, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(units=64, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(units=10, activation='softmax'),])
  return model
