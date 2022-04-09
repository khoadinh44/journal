from load_cases import get_data
from FaceNet_predict import FaceNetOneShotRecognitor
from preprocessing.utils import invert_one_hot, load_table_10_spe, recall_m, precision_m, f1_m, to_one_hot
from network.nn import CNN_C
from src.model import CNN_C_trip
from load_cases import get_data
from train_routines.triplet_loss import train, parse_opt
from train_routines.center_loss import train_center_loss
from train_routines.triplet_center_loss import train_triplet_center_loss
from train_routines.xentropy import train_xentropy

from sklearn.utils import shuffle
from scipy.spatial.distance import cosine, euclidean
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def main(opt):
  print(color.GREEN + '\n\n\t *************START*************\n\n' + color.END)
  
  print('\n loading data...')
  X_train, X_test, y_train, y_test = get_data(opt)
  y_train = invert_one_hot(y_train)
  y_test = invert_one_hot(y_test)
  print(f' Training data shape: {X_train.shape}  Training label shape: {y_train.shape}')
  print(f' Testing data shape: {X_test.shape}  Testing label shape: {y_test.shape}')

  train_embs, test_embs = train(opt, X_train, y_train, X_test, y_test, CNN_C_trip) 

  print('\n Saving embedding phase...')   
  this_acc = []
  y_pred_all = []
  y_pred_Lo_Co = []
  y_pred_SVM_Ran = []
  
  l = 0
  for each_ML in ['SVM', 'RandomForestClassifier', 'LogisticRegression', 'GaussianNB', 'euclidean', 'cosine']:
    model = FaceNetOneShotRecognitor(opt, X_train, y_train) 
    y_pred = model.predict(test_embs=test_embs, train_embs=train_embs, threshold=1, ML_method=each_ML)

    y_pred_onehot = to_one_hot(y_pred)
    if y_pred_all == []:
      y_pred_all = y_pred_onehot
    else:
      y_pred_all = y_pred_all + y_pred_onehot

    l += 1

    if each_ML == 'SVM' or each_ML == 'RandomForestClassifier':
      if y_pred_SVM_Ran == []:
        y_pred_SVM_Ran = y_pred_onehot
      else:
        y_pred_SVM_Ran = y_pred_SVM_Ran + y_pred_onehot
      
    if each_ML == 'LogisticRegression' or 'cosine':
      if y_pred_Lo_Co == []:
        y_pred_Lo_Co = y_pred_onehot
      else:
        y_pred_Lo_Co = y_pred_Lo_Co + y_pred_onehot

    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/{each_ML}.npy', 'wb') as f:
      np.save(f, y_pred)
    with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/{each_ML}.txt', 'wb') as f:
      for i in list(y_pred):
        f.write(f' {str(i)}')
        f.write(', ')

    acc = accuracy_score(y_test, y_pred)
    print(f'\n--------------Test accuracy: {acc} with the {each_ML} method--------------')
    
  y_pred_all = y_pred_all.astype(np.float32) / l
  y_pred_all = np.argmax(y_pred_all, axis=1)
  acc_all = accuracy_score(y_pred_all, y_pred)
  print(f'\n--------------Ensemble for all: {acc_all}--------------')

  with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/ensemble.npy', 'wb') as f:
    np.save(f, y_pred_all)
  with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/ensemble.txt', 'wb') as f:
    for i in list(y_pred_all):
      f.write(f' {str(i)}')
      f.write(',')

  with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/true_label.npy', 'wb') as f:
    np.save(f, y_pred)
  with open(f'/content/drive/Shareddrives/newpro112233/signal_machine/output_triplet_loss/true_label.txt', 'wb') as f:
    for i in list(y_pred):
      f.write(f' {str(i)}')
      f.write(', ')

  y_pred_Lo_Co = y_pred_Lo_Co.astype(np.float32) / 2
  y_pred_Lo_Co = np.argmax(y_pred_Lo_Co, axis=1)
  acc_Lo_Co = accuracy_score(y_pred_Lo_Co, y_pred)
  print(f'\n--------------Ensemble for LogisticRegression and cosine: {acc_Lo_Co}--------------')

  y_pred_SVM_Ran = y_pred_SVM_Ran.astype(np.float32) / 2
  y_pred_SVM_Ran = np.argmax(y_pred_SVM_Ran, axis=1)
  acc_SVM_Ran = accuracy_score(y_pred_SVM_Ran, y_pred)
  print(f'\n--------------Ensemble for SVM and RandomForestClassifier: {acc_SVM_Ran}--------------')

if __name__ == '__main__':
  opt = parse_opt()
  main(opt)
