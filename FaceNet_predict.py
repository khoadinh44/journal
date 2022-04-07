from scipy.spatial.distance import cosine, euclidean
from load_cases import get_data
from train_routines.triplet_loss import parse_opt
import angular_grad
import tensorflow as tf
import glob 
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from preprocessing.utils import invert_one_hot

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

opt = parse_opt()

class FaceNetOneShotRecognitor(object):
    def __init__(self, opt, X_train_all, y_train_all):
        self.opt         = opt
        self.X_train_all, self.y_train_all = X_train_all, y_train_all
        self.df_train = pd.DataFrame(columns=['all_train_data', 'ID', 'name'])

    def __calc_embs(self, input_data):
        pd = []
        for start in tqdm(range(0, len(input_data), self.opt.batch_size)):
            embeddings = self.model(input_data[start: start+self.opt.batch_size])
            pd.append(tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10))
        return np.array(pd)
    
    def __calc_emb_test(self, input_data):
        pd = []
        if len(input_data) == 1:
            embeddings = self.model(input_data)
            pd.append(tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10))
        elif len(input_data) > 1:
            pd = self.__calc_embs(input_data)
        return np.array(pd)
    
    def loading(self, input_):
        all_data = []
        for i in input_:
            all_data.append(i)
        return np.array(all_data)
      
    def predict(self, test_embs, train_embs, threshold=1.1, ML_method=None):
        print('\n Test embs: ', test_embs.shape)
        print(' Train embs: ', train_embs.shape)
        list_label = {}

        for ID, (train_data, train_label) in enumerate(zip(self.X_train_all, self.y_train_all)):
            self.df_train.loc[len(self.df_train)] = [train_data, ID, train_label]

        if ML_method == 'euclidean' or ML_method == 'cosine':
          for i in range(test_embs.shape[0]):
              distances = []
              for j in range(train_embs.shape[0]):
                  # the min of clustering
                  if ML_method == 'euclidean':
                    distances.append(euclidean(test_embs[i].reshape(-1), train_embs[j]))
                  elif ML_method == 'cosine':
                    distances.append(cosine(test_embs[i].reshape(-1), train_embs[j]))
              if np.min(distances) > threshold:
                  list_label[i] = 100  # 100 is represented for unknown object
              else:
                  res = np.argsort(distances)[0]  # this ID
                  list_label[i] = res

          if len(list_label) > 0:
              for idx in list_label:
                  if list_label[idx] != 100:
                      name = self.df_train[( self.df_train['ID'] == list_label[idx] )].name.iloc[0]
                      list_label[idx] = name

          list_label = list(list_label.values())
        
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
          model.fit(train_embs, train_label)
          list_label = model.predict(test_embs)
          
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
