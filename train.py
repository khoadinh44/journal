import tensorflow as tf
import numpy as np
from network.dnn import network
from sklearn.model_selection import train_test_split
from load_data import Normal_0_reshape, Normal_0_name,\
                      B007_0_reshape, B007_0_name,\
                      IR007_0_reshape, IR007_0_name,\
                      OR007_3_0_reshape, OR007_3_0_name,\
                      OR007_6_0_reshape, OR007_6_0_name,\
                      OR007_12_0_reshape, OR007_12_0_name

data = np.concatenate((Normal_0_reshape, B007_0_reshape, IR007_0_reshape, OR007_3_0_reshape, OR007_6_0_reshape, OR007_12_0_reshape))
label = np.concatenate((Normal_0_name, B007_0_name, IR007_0_name, OR007_3_0_name, OR007_6_0_name, OR007_12_0_name))

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42, shuffle=True)
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

train(X_train, y_train, X_test, y_test, network, 200, 32, True, 'model.h5')
