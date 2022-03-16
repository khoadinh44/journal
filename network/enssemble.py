from tensorflow import keras
# import keras
import numpy as np
from preprocessing.utils import accuracy_m
from network.resnet import resnet18, resnet34
from network.nn import DNN_A, DNN_B, CNN_A, CNN_B, CNN_C 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from preprocessing.utils import invert_one_hot, convert_one_hot
from preprocessing.utils import use_denoise
from preprocessing.denoise_signal import Fourier


def semble_transfer(opt, X_test, y_test, X_train, y_train):
  y_pred = np.zeros(shape=y_test.shape)
  l = 0
  
  for name in opt.model_names:
    all_path = opt.model_dir + name 
    if name == 'CNN_B':
      inputs = keras.Input(shape=(360, 360, 3))
      outputs = resnet18(inputs)
      model = keras.Model(inputs, outputs)
      curr_y_pred = model.predict(X_test)
      keras.backend.clear_session()
      np.save(opt.model_dir + name + '.npy', curr_y_pred)
    
      y_pred += curr_y_pred
      l += 1

    
    if name == 'CNN_C':
      model = CNN_C(opt)
      model.load_weights(all_path)
      curr_y_pred = model.predict(X_test)
      keras.backend.clear_session()
      np.save(opt.model_dir + name + '.npy', curr_y_pred)
    
      y_pred += curr_y_pred
      l += 1

  model = SVC(kernel='rbf', probability=True)
  y_train = invert_one_hot(y_train)
  # X_train = use_denoise(X_train, Fourier)
  # X_test = use_denoise(X_test, Fourier)

  model.fit(X_train, y_train)
  test_predictions = model.predict(X_test)
  test_predictions = convert_one_hot(test_predictions)
  y_pred += test_predictions
  l += 1
    
  y_pred = y_pred / l
  print('Test accuracy: ', accuracy_m(y_test, y_pred))
    
