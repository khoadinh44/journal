from src.data import get_dataset
from src.params import Params
from faceNet import opt

class FaceNetOneShotRecognitor(object):
    def __init__(self, opt, X_train_all, y_train_all):
        self.path_weight = opt.ckpt_dir
        self.params      = Params(opt.params_dir)
        
        # INITIALIZE MODELS
        self.params      = Params(args.params_dir)
        self.model       = face_model(self.params)
        self.optimizer = angular_grad.AngularGrad()
        self.checkpoint  = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, train_steps=tf.Variable(0,dtype=tf.int64),
                                               valid_steps=tf.Variable(0,dtype=tf.int64), epoch=tf.Variable(0, dtype=tf.int64))
        self.ckptmanager = tf.train.CheckpointManager(self.checkpoint, self.path_weight, 3)
            
        self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
        print(f'\nRestored from Checkpoint : {self.ckptmanager.latest_checkpoint}\n')

        self.graph = tf.compat.v1.get_default_graph()
        
        self.train_data = glob.glob(path_train)
        self.nb_classes = len(self.train_paths)
        
        self.train_dataset, self.train_samples = get_dataset(X_train_all, y_train_all, self.params, 'train')
        self.df_train = pd.DataFrame(columns=['image', 'label', 'name'])
        
    def __l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output
    
    def __calc_embs(self, input_data, batch_size=32):
        pd = []
        for start in tqdm(range(0, len(filepaths), batch_size)):
            embeddings = self.model(input_data[start: start+batch_size])
            pd.append(tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10))
        return np.array(pd)
    
    def __calc_emb_test(self, input_data):
        pd = []
        if (int(input_data.shape[0]) == 1):
            embeddings = self.model(input_data)
        elif (int(input_data.shape[0]) > 1):
            for start in tqdm(range(0, len(filepaths), batch_size)):
                embeddings = self.model(input_data[start: start+batch_size])
        pd.append(tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10))
        return np.array(pd)
      
    def train_or_load(self, batch_size, cons=True):
        for ID, (train_data, train_label) in enumerate(self.train_dataset):
            for image in images:
                self.df_train.loc[self.train_samples] = [train_data, ID, train_label]

        # TRAINING
        label2idx = []
        for i in tqdm(range(self.train_samples)):
            label2idx.append(np.asarray(self.df_train[self.df_train.label == i].index))
        if cons:
            train_embs = self.__calc_embs(self.df_train.image, batch_size)
            np.save(opt.emb_dir, train_embs)
        else:
            train_embs = np.load(opt.emb_dir, allow_pickle=True)
        # print(train_embs.shape)
        train_embs = np.concatenate(train_embs)

        return train_embs, label2idx
      
    def predict(self, test_data, train_embs, label2idx, threshold=0.15):
        test_embs = self.__calc_emb_test(test_data)
#         test_embs = np.concatenate(test_embs)
        each_label = {}
        for i in range(test_embs.shape[0]):
            distances = []
            for j in range(len(self.train_paths)):
                # the min of clustering
                distances.append(np.min([euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
                # distances.append(np.min([cosine(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))

            if np.min(distances) > threshold:
                each_label[i] = 'Unknown'
            else:
                res = np.argsort(distances)[0]
                each_label[i] = res

        names = []
        if len(each_label) > 0:
            for idx in each_label:
                if each_label[idx] != 'Unknown':
                    name = self.df_train[(self.df_train['label'] == each_label[idx])].name.iloc[0]
                    name = name.split("/")[-1]
                    each_label[idx] = name

        return people

if __name__ == '__main__':
    X_train_all, X_test, y_train_all, y_test = get_data(opt)
    model = FaceNetOneShotRecognitor(opt, X_train_all, y_train_all)
    train_embs, label2idx = model.train_or_load(batch_size=64, cons=True)
    
    params = Params(opt.params_dir)
    self.valid_dataset, self.valid_samples = get_dataset(X_test, y_test, params, 'val')
    people = model.predict(faces=faces, train_embs=train_embs, label2idx=label2idx)
