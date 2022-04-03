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
from preprocessing.utils import invert_one_hot

opt = parse_opt()

class FaceNetOneShotRecognitor(object):
    def __init__(self, opt, X_train_all, y_train_all):
        self.X_train_all, self.y_train_all = X_train_all, y_train_all
        self.path_weight = opt.ckpt_dir
        self.opt         = opt
        self.params      = Params(opt.params_dir)
        
        # INITIALIZE MODELS
        self.model       = face_model(opt)
        self.optimizer   = angular_grad.AngularGrad()
        self.checkpoint  = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, train_steps=tf.Variable(0, dtype=tf.int64),
                                               valid_steps=tf.Variable(0, dtype=tf.int64), epoch=tf.Variable(0, dtype=tf.int64))
        self.ckptmanager = tf.train.CheckpointManager(self.checkpoint, self.path_weight, 3)
            
        self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
        print(f'\nRestored from Checkpoint : {self.ckptmanager.latest_checkpoint}\n')

        self.graph = tf.compat.v1.get_default_graph()
        
        if len(np.array(self.y_train_all).shape) > 1:
            self.y_train_all = invert_one_hot(self.y_train_all)
        self.train_dataset, self.train_samples = get_dataset(self.X_train_all, self.y_train_all, self.params, 'train')
        self.df_train = pd.DataFrame(columns=['all_train_data', 'ID', 'name'])
        
    def __l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

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
      
    def train_or_load(self, cons=True):        
        for ID, (train_data, train_label) in enumerate(zip(self.X_train_all, self.y_train_all)):
            self.df_train.loc[len(self.df_train)] = [train_data, ID, train_label]

        if cons:
            train_embs = self.__calc_embs(self.X_train_all)
            np.save(opt.emb_dir, train_embs)
        else:
            train_embs = np.load(opt.emb_dir, allow_pickle=True)
        train_embs = np.concatenate(train_embs)

        return train_embs
      
    def predict(self, test_data, train_embs, threshold=1.1):
        test_embs = self.__calc_emb_test(test_data)
        test_embs = np.concatenate(test_embs)
        print('\n Test embs: ', test_embs.shape)
        print(' Train embs: ', train_embs.shape)
        list_label = {}

        for i in range(test_embs.shape[0]):
            distances = []
            for j in range(self.train_samples):
                # the min of clustering
                distances.append(euclidean(test_embs[i].reshape(-1), train_embs[j]))
                # distances.append(np.min([cosine(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
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
