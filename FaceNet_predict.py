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
        
        self.train_dataset, self.train_samples = get_dataset(X_train_all, y_train_all, self.params, 'train')
        self.df_train = pd.DataFrame(columns=['all_train_data', 'label', 'name'])
        
    def __l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output
    
    def load_data(self, input_data):
      all_data = []
      for i in input_data:
        all_data.append(i)
      return np.array(all_data)

    def __calc_embs(self, input_data):
        pd = []
        for start in tqdm(range(0, len(input_data), self.opt.batch_size)):
            embeddings = self.model(self.load_data(input_data[start: start+self.opt.batch_size]))
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
        for ID, (train_data, train_label) in enumerate(self.train_dataset):
            self.df_train.loc[len(self.df_train)] = [np.squeeze(train_data.numpy()), ID, train_label.numpy()[0]]

        # TRAINING
        label2idx = []
        for i in tqdm(range(self.train_samples)):
            label2idx.append(np.asarray(self.df_train[self.df_train.label == i].index))
        if cons:
            train_embs = self.__calc_embs(self.df_train.all_train_data)
            np.save(opt.emb_dir, train_embs)
        else:
            train_embs = np.load(opt.emb_dir, allow_pickle=True)
        # print(train_embs.shape)
        train_embs = np.concatenate(train_embs)

        return train_embs, label2idx
      
    def predict(self, test_data, train_embs, label2idx, threshold=1.7):
        test_embs = self.__calc_emb_test(test_data)
        test_embs = np.concatenate(test_embs)
        print('\ntest_embs: ', test_embs.shape)
        each_label = {}

        for i in range(test_embs.shape[0]):
            distances = []
            for j in range(self.train_samples):
                # the min of clustering
                distances.append(np.min([euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
                # distances.append(np.min([cosine(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
            if np.min(distances) > threshold:
                each_label[i] = 100  # 100 is represented for unknown object
            else:
                res = np.argsort(distances)[0]
                each_label[i] = res

        names = []
        if len(each_label) > 0:
            for idx in each_label:
                if each_label[idx] != 100:
                    name = self.df_train[(self.df_train['label'] == each_label[idx])].name.iloc[0]
                    each_label[idx] = name

        each_label = list(each_label.values())
        return each_label

if __name__ == '__main__':
    X_train_all, X_test, y_train_all, y_test = get_data(opt)
    y_test = invert_one_hot(y_test)

    print('Shape of train data: ', X_train_all.shape)
    print('Shape of test data: ', X_test.shape)

    model = FaceNetOneShotRecognitor(opt, X_train_all, y_train_all)
    train_embs, label2idx = model.train_or_load(cons=False)
    
    params = Params(opt.params_dir)
    y_pred = model.predict(test_data=X_test, train_embs=train_embs, label2idx=label2idx)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n--------------Test accuracy: {acc}----------------')
