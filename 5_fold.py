from faceNet import Trainer, parse_opt
from FaceNet_predict import FaceNetOneShotRecognitor
from load_data import Healthy, Outer_ring_damage, Inner_ring_damage 
from preprocessing.utils import invert_one_hot, load_table_10_spe
from src.params import Params

from src.data import get_dataset
from scipy.spatial.distance import cosine, euclidean
from load_cases import get_data
from src.params import Params
from faceNet import parse_opt
from src.model  import face_model
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

opt = parse_opt()

print('\t\t\t Loading labels...')
accuracy = []

for i in range(len(Healthy)):
  X_train_Healthy = Healthy[i: i+3]
  if len(X_train_Healthy) != 3:
    break
  y_train_Healthy = [0]*len(X_train_Healthy)
  X_train_Healthy, y_train_Healthy = load_table_10_spe(X_train_Healthy, y_train_Healthy)
  print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
  
  X_train_Outer_ring_damage = Outer_ring_damage[i: i+3]
  y_train_Outer_ring_damage = [1]*len(X_train_Outer_ring_damage)
  X_train_Outer_ring_damage, y_train_Outer_ring_damage = load_table_10_spe(X_train_Outer_ring_damage, y_train_Outer_ring_damage)
  print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
  
  X_train_Inner_ring_damage = Inner_ring_damage[i: i+3]
  y_train_Inner_ring_damage = [2]*len(X_train_Inner_ring_damage)
  X_train_Inner_ring_damage, y_train_Inner_ring_damage = load_table_10_spe(X_train_Inner_ring_damage, y_train_Inner_ring_damage)
  print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
  
  X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
  y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))
  print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
  
  print('\n------------------------------------------------')

  h = [a for a in range(len(Healthy)) if a not in range(i, i+3)]
  X_test_Healthy = Healthy[h]
  y_test_Healthy = [0]*len(X_test_Healthy)
  X_test_Healthy, y_test_Healthy = load_table_10_spe(X_test_Healthy, y_test_Healthy)
  print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
  
  k = [a for a in range(len(Outer_ring_damage)) if a not in range(i, i+3)]
  X_test_Outer_ring_damage = Outer_ring_damage[k]
  y_test_Outer_ring_damage = [1]*len(X_test_Outer_ring_damage)
  X_test_Outer_ring_damage, y_test_Outer_ring_damage = load_table_10_spe(X_test_Outer_ring_damage, y_test_Outer_ring_damage)
  print(f'\n Shape of the Outer ring damage test data and label: {X_test_Outer_ring_damage.shape}, {y_test_Outer_ring_damage.shape}')
  
  l = [a for a in range(len(Inner_ring_damage)) if a not in range(i, i+3)]
  X_test_Inner_ring_damage = Inner_ring_damage[l]
  y_test_Inner_ring_damage = [2]*len(X_test_Inner_ring_damage)
  X_test_Inner_ring_damage, y_test_Inner_ring_damage = load_table_10_spe(X_test_Inner_ring_damage, y_test_Inner_ring_damage)
  print(f'\n Shape of the Inner ring damage test data and label: {X_test_Inner_ring_damage.shape}, {y_test_Inner_ring_damage.shape}')
  
  X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
  y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
  print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
  
  print('\n Train phase...')
  trainer = Trainer(opt, X_train, X_test, y_train, y_test)
  for i in range(opt.epoch):
      trainer.train(i)
  
  print('\n Test phase...')
  model = FaceNetOneShotRecognitor(opt, X_train, y_train)
  train_embs, label2idx = model.train_or_load(cons=True)
  
  params = Params(opt.params_dir)
  this_acc = []
  for thres in opt.threshold:
    y_pred = model.predict(test_data=X_test, train_embs=train_embs, label2idx=label2idx, threshold=thres)
    acc = accuracy_score(y_test, y_pred)
    this_acc.append(acc)
    print(f'\n--------------Test accuracy: {acc} in the threshold of {thres}----------------')
  
  accuracy.append(max(this_acc))

print('\n FINISH!')
print('Test accuracy: ', np.mean(accuracy))
