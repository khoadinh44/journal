from network.cnn import network 
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from load_cnn import merge_data, label

X_train, X_test, y_train, y_test = train_test_split(merge_data, label, test_size=0.1, random_state=42, shuffle=True)
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

train(X_train, y_train, X_test, y_test, merge_network, 200, 32, True, 'model.h5')
