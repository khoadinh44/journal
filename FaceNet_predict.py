from scipy.spatial.distance import cosine, euclidean
from load_cases import get_data
from train import parse_opt
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing.utils import invert_one_hot, onehot

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

opt = parse_opt()

class FaceNetOneShotRecognitor(object):
    def __init__(self, opt, X_train_all, y_train_all, X_test, y_test):
        self.opt         = opt
        self.X_train_all, self.y_train_all = X_train_all, y_train_all
        self.X_test, self.y_test = X_test, y_test

        self.df_train = pd.DataFrame(columns=['all_train_data', 'ID', 'name'])

    def __calc_embs(self, input_data):
        pd = []
        for start in tqdm(range(0, len(input_data), self.opt.batch_size)):
            embeddings = self.model(input_data[start: start+self.opt.batch_size])
            # pd.append(tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10))
            pd.append(embeddings)
        return np.array(pd)
    
    def __calc_emb_test(self, input_data):
        pd = []
        if len(input_data) == 1:
            embeddings = self.model(input_data)
            # pd.append(tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10))
            pd.append(embeddings)
        elif len(input_data) > 1:
            pd = self.__calc_embs(input_data)
        return np.array(pd)
    
    def loading(self, input_):
        all_data = []
        for i in input_:
            all_data.append(i)
        return np.array(all_data)
      
    def predict(self, test_embs, train_embs, ML_method=None, emb=True, use_mean=True):
        print('\n Test embs: ', test_embs.shape)
        print(' Train embs: ', train_embs.shape)
        
        if use_mean:
            list_label = []
        else:
            list_label = {}

        for ID, (train_data, train_label) in enumerate(zip(self.X_train_all, self.y_train_all)):
            self.df_train.loc[len(self.df_train)] = [train_data, ID, train_label]
        
        mean_class_0 = np.mean(train_embs[self.y_train_all==0], axis=0)
        var_class_0  = np.var(train_embs[self.y_train_all==0], axis=0)
        emb_class_0  = np.concatenate((mean_class_0, var_class_0))
        
        mean_class_1 = np.mean(train_embs[self.y_train_all==1], axis=0)
        var_class_1  = np.var(train_embs[self.y_train_all==1], axis=0)
        emb_class_1  = np.concatenate((mean_class_1, var_class_1))
        
        mean_class_2 = np.mean(train_embs[self.y_train_all==2], axis=0)
        var_class_2  = np.var(train_embs[self.y_train_all==2], axis=0)
        emb_class_2  = np.concatenate((mean_class_2, var_class_2))
        
        emb_class_all = [emb_class_0, emb_class_1, emb_class_2]

        if ML_method == 'euclidean' or ML_method == 'cosine':
          for i in range(test_embs.shape[0]):
              distances = []

              if use_mean:
                  for emb_class in emb_class_all:
                      test_emb = test_embs[i].reshape(-1)
                      
                      half = emb_class.shape[0]//2
                      mean_each, var_each = emb_class[ :half], emb_class[half: ]
                      test_combi = np.concatenate((np.expand_dims(mean_each, axis=0), np.expand_dims(test_emb, axis=0)))
                      test_var = np.var(test_combi, axis=0)

                      # emb_test_each = np.concatenate((test_emb, test_var))
                      if ML_method == 'euclidean':
                          distances.append(np.sqrt(euclidean(mean_each, test_emb)**2 + euclidean(var_each, test_var)**2)) # append one value
                      elif ML_method == 'cosine':
                          distances.append(cosine(mean_each, test_emb) + cosine(var_each, test_var))
                  list_label.append(np.argsort(distances)[0])
              else:
                  for j in range(train_embs.shape[0]):
                      if ML_method == 'euclidean':
                        distances.append(euclidean(test_embs[i].reshape(-1), train_embs[j])) # append one value
                      elif ML_method == 'cosine':
                        distances.append(cosine(test_embs[i].reshape(-1), train_embs[j]))
                  res = np.argsort(distances)[0]  
                  # res = np.exp(x)/sum(np.exp(x))
                  # list_label.append(res.tolist())
                  list_label[i] = res

          if use_mean == False:
              if len(list_label) > 0:
                  for idx in list_label:
                      name = self.df_train[( self.df_train['ID'] == list_label[idx] )].name.iloc[0]
                      list_label[idx] = name
              list_label = list(list_label.values())

          list_label = onehot(list_label)
        else:
          train_label = self.df_train['name']
          train_label = self.loading(train_label)
          if ML_method == 'SVM':
            model = SVC(kernel='rbf', probability=True)
          elif ML_method == 'RandomForestClassifier':
            model = RandomForestClassifier(n_estimators= 300, max_features = "sqrt", n_jobs = -1, random_state = 38)
          elif ML_method == 'LogisticRegression':     
            model = LogisticRegression(random_state=1)
          elif ML_method == 'GaussianNB':
            model = GaussianNB()
          elif ML_method == 'KNN':     
            model = KNeighborsClassifier(n_neighbors=3)
          elif ML_method == 'BT':
            model = GradientBoostingClassifier()
            
          if emb:
            model.fit(train_embs, train_label)
            list_label = model.predict_proba(test_embs)
          else:
            model.fit(self.X_train_all, self.y_train_all)
            list_label = model.predict_proba(test_embs)
          
        return list_label

if __name__ == '__main__':
    X_train_all, X_test, y_train_all, y_test = get_data(opt)
    y_test = invert_one_hot(y_test)

    print('Shape of train data: ', X_train_all.shape)
    print('Shape of test data: ', X_test.shape)

    model = FaceNetOneShotRecognitor(opt, X_train_all, y_train_all)
    train_embs = model.train_or_load(cons=True)
    
    params = Params(opt.params_dir)
    y_pred = model.predict(test_data=X_test, train_embs=train_embs)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n--------------Test accuracy: {acc}----------------')
