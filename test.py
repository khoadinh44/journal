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

t_data = merge_data[:1, :]
t_data_A, t_data_B = t_data[:, :623], t_data[:, 623:]
model = merge_network()
model.load_weights('model.h5')

y_pred = model.predict((t_data_A, t_data_B))
print(np.argmax(y_pred))
