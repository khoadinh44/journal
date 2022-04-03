from faceNet import Trainer, parse_opt
from FaceNet_predict import FaceNetOneShotRecognitor
from load_data import Healthy, Outer_ring_damage, Inner_ring_damage 
from preprocessing.utils import invert_one_hot, load_table_10_spe, recall_m, precision_m, f1_m, to_one_hot
from src.params import Params
from network.nn import CNN_C
from src.data import get_dataset
from load_cases import get_data
from src.params import Params
from faceNet import parse_opt
from src.model  import face_model

from sklearn.utils import shuffle
from scipy.spatial.distance import cosine, euclidean
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

opt = parse_opt()

print('\t\t\t Loading labels...')
Healthy_label           = [0]*len(Healthy)
Outer_ring_damage_label = [1]*len(Outer_ring_damage)
Inner_ring_damage_label = [2]*len(Inner_ring_damage)

Healthy, Healthy_label = load_table_10_spe(Healthy, Healthy_label)
Outer_ring_damage, Outer_ring_damage_label = load_table_10_spe(Outer_ring_damage, Outer_ring_damage_label)
Inner_ring_damage, Inner_ring_damage_label = load_table_10_spe(Inner_ring_damage, Inner_ring_damage_label)

np.random.seed(0)
Healthy, Healthy_label = shuffle(Healthy, Healthy_label, random_state=0)
Outer_ring_damage, Outer_ring_damage_label = shuffle(Outer_ring_damage, Outer_ring_damage_label, random_state=0)
Inner_ring_damage, Inner_ring_damage_label = shuffle(Inner_ring_damage, Inner_ring_damage_label, random_state=0)

print('\n\n\t *************START*************\n\n')
accuracy = []

for i in range(5):
  distance_Healthy = int(0.6*len(Healthy))
  start_Healthy    = int(0.2*i*len(Healthy))
  X_train_Healthy = Healthy[start_Healthy: start_Healthy+distance_Healthy]
  if len(X_train_Healthy) < distance_Healthy:
    break
  y_train_Healthy = Healthy_label[start_Healthy: start_Healthy+distance_Healthy]
  print(f'\n Shape of the Health train data and label: {X_train_Healthy.shape}, {y_train_Healthy.shape}')
  
  distance_Outer_ring_damage = int(0.6*len(Outer_ring_damage))
  start_Outer_ring_damage    = int(0.2*i*len(Outer_ring_damage))
  X_train_Outer_ring_damage, y_train_Outer_ring_damage = Outer_ring_damage[start_Outer_ring_damage: start_Outer_ring_damage+distance_Outer_ring_damage], Outer_ring_damage_label[start_Outer_ring_damage: start_Outer_ring_damage + distance_Outer_ring_damage]
  print(f'\n Shape of the Outer ring damage train data and label: {X_train_Outer_ring_damage.shape}, {y_train_Outer_ring_damage.shape}')
  
  distance_Inner_ring_damage = int(0.6*len(Inner_ring_damage))
  start_Inner_ring_damage    = int(0.2*i*len(Inner_ring_damage))
  X_train_Inner_ring_damage, y_train_Inner_ring_damage = Inner_ring_damage[start_Inner_ring_damage: start_Inner_ring_damage + distance_Inner_ring_damage], Inner_ring_damage_label[start_Inner_ring_damage: start_Inner_ring_damage + distance_Inner_ring_damage]
  print(f'\n Shape of the Inner ring damage train data and label: {X_train_Inner_ring_damage.shape}, {y_train_Inner_ring_damage.shape}')
  
  X_train = np.concatenate((X_train_Healthy, X_train_Outer_ring_damage, X_train_Inner_ring_damage))
  y_train = np.concatenate((y_train_Healthy, y_train_Outer_ring_damage, y_train_Inner_ring_damage))
  print(f'\n Shape of train data: {X_train.shape}, {y_train.shape}')
  
  print('\n------------------------------------------------')

  h = [a for a in range(len(Healthy)) if a not in range(start_Healthy, start_Healthy+distance_Healthy)]
  X_test_Healthy = Healthy[h]
  y_test_Healthy = Healthy_label[h]
  print(f'\n Shape of the Health test data and label: {X_test_Healthy.shape}, {y_test_Healthy.shape}')
  
  k = [a for a in range(len(Outer_ring_damage)) if a not in range(start_Outer_ring_damage, start_Outer_ring_damage+distance_Outer_ring_damage)]
  X_test_Outer_ring_damage = Outer_ring_damage[h]
  y_test_Outer_ring_damage = Outer_ring_damage_label[h]
  print(f'\n Shape of the Outer ring damage test data and label: {X_test_Outer_ring_damage.shape}, {y_test_Outer_ring_damage.shape}')
  
  l = [a for a in range(len(Inner_ring_damage)) if a not in range(start_Inner_ring_damage, start_Inner_ring_damage+distance_Inner_ring_damage)]
  X_test_Inner_ring_damage = Inner_ring_damage[h]
  y_test_Inner_ring_damage = Inner_ring_damage_label[h]
  print(f'\n Shape of the Inner ring damage test data and label: {X_test_Inner_ring_damage.shape}, {y_test_Inner_ring_damage.shape}')
  
  X_test = np.concatenate((X_test_Healthy, X_test_Outer_ring_damage, X_test_Inner_ring_damage))
  y_test = np.concatenate((y_test_Healthy, y_test_Outer_ring_damage, y_test_Inner_ring_damage))
  print(f'\n Shape of test data: {X_test.shape}, {y_test.shape}')
  print('\n------------------------------------------------')

  if opt.faceNet:
    print('\n Train phase...')
    X_test = np.expand_dims(X_test, axis=-1).astype(np.float32)
    X_train = np.expand_dims(X_train, axis=-1).astype(np.float32)

    trainer = Trainer(opt, X_train, X_test, y_train, y_test)
    for i in range(opt.epoch):
        trainer.train(i)
    
    print('\n Saving embedding phase...')
    model = FaceNetOneShotRecognitor(opt, X_train, y_train)
    train_embs = model.train_or_load(cons=True)
    
    params = Params(opt.params_dir)
    this_acc = []
    for thres in opt.threshold:
      print('\n Predict phase...')
      y_pred = model.predict(test_data=X_test, train_embs=train_embs, threshold=thres)
      acc = accuracy_score(y_test, y_pred)
      this_acc.append(acc)
      print(f'\n--------------Test accuracy: {acc} in the threshold of {thres}----------------')
    
    accuracy.append(max(this_acc))
    print(f'\n\t\t********* FINISHING ROUND {i}*********\n\n\n')
  else:
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)
    print('\n\t\t\t Load model...')
    model = CNN_C(opt)
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m]) # loss='mse'

    model.summary()
    history = model.fit(X_train, y_train,
                        epochs     = opt.epoch,
                        batch_size = opt.batch_size,
                        validation_data=(X_test, y_test),)
    _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = model.evaluate(X_test, y_test, verbose=0)
    accuracy.append(test_acc)

print('\n FINISH!')
print('Test accuracy: ', np.mean(accuracy))
