from tensorflow import keras
from preprocessing.utils import accuracy_m

def semble_transfer(opt, X_test, y_test):
  y_pred = np.zeros(shape=y_test.shape)
  l = 0
  
  for name in opt.model_names:
    all_path = opt.model_dir + name + '.hdf5'
    curr_y_pred = model.predict(X_test)
    keras.backend.clear_session()
    np.save(opt.model_dir + name + '.npy', curr_y_pred)
    
    y_pred += curr_y_pred
    l += 1
    keras.backend.clear_session()
    
  y_pred = y_pred / l
  return accuracy_m(y_test, y_pred)
    
