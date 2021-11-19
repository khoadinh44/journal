import tensorflow as tf
import numpy as np
from network.dnn import network, merge_network
from sklearn.model_selection import train_test_split
from load_data import Normal_0_reshape,\
                      B007_0_reshape,\
                      IR007_0_reshape,\
                      OR007_3_0_reshape,\
                      OR007_6_0_reshape,\
                      OR007_12_0_reshape,\
                      merge_data, label

# data = np.concatenate((Normal_0_reshape, B007_0_reshape, IR007_0_reshape, OR007_3_0_reshape, OR007_6_0_reshape, OR007_12_0_reshape))

X_train, X_test, y_train, y_test = train_test_split(merge_data, label, test_size=0.1, random_state=42, shuffle=True)
X_train_A, X_train_B = X_train[:, :623], X_train[:, 623:]
X_test_A, X_test_B = X_test[:, :623], X_test[:, 623:]
# active GPU
# tf.debugging.set_log_device_placement(True)

# Load model------------------------------------------------------------------------------------
def train(data=None, labels=None,\
          val_data=None, val_labels=None,\
          network=None, num_epochs=20,\
          batch_size=32, show_metric=True, name_saver=None):
  model = network()
  model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
  history = model.fit(data, labels, epochs=num_epochs,
                    validation_data=(val_data, val_labels))
  model.save(name_saver)

train((X_train_A, X_train_B), y_train, (X_test_A, X_test_B), y_test, merge_network, 200, 32, True, 'model.h5')
