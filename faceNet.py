import os
import argparse
import datetime
import tensorflow as tf
from progressbar import *
import angular_grad
from load_cases import get_data
from src.params import Params
from src.model  import face_model
from src.data   import get_dataset
from src.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, adapted_triplet_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Trainer():
    def __init__(self, opt, X_train_all, X_test, y_train_all, y_test):
        
        self.params      = Params(opt.params_dir)
        self.valid       = 1 if opt.validate == '1' else 0
        self.model       = face_model(opt)
        
        
        # self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.params.learning_rate,
        #                                                                   decay_steps=10000, decay_rate=0.96, staircase=True)
        # self.optimizer   = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)
        self.optimizer = angular_grad.AngularGrad()
        self.checkpoint  = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, train_steps=tf.Variable(0,dtype=tf.int64),
                                               valid_steps=tf.Variable(0,dtype=tf.int64), epoch=tf.Variable(0, dtype=tf.int64))
        self.ckptmanager = tf.train.CheckpointManager(self.checkpoint, opt.ckpt_dir, 3)
        
        if self.params.triplet_strategy == "batch_all":
            self.loss = batch_all_triplet_loss
            
        elif self.params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss
            
        elif self.params.triplet_strategy == "batch_adaptive":
            self.loss = adapted_triplet_loss
            
        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
        opt.log_dir += current_time + '/train/'
        self.train_summary_writer = tf.summary.create_file_writer(opt.log_dir)
            
        if opt.restore == '1':
            self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
            print(f'\nRestored from Checkpoint : {self.ckptmanager.latest_checkpoint}\n')
        
        else:
            print('\nIntializing from scratch\n')
        
        
        print(f'Shape of training data: {X_train_all.shape}\n')
        print(f'Shape of testing data: {X_test.shape}\n')
       
        self.train_dataset, self.train_samples = get_dataset(X_train_all, y_train_all, self.params, 'train')
        
        if self.valid:
            self.valid_dataset, self.valid_samples = get_dataset(X_test, y_test, self.params, 'val')
        
        
    def __call__(self, epoch):
        for i in range(epoch):
            self.train(i)
            if self.valid:
                self.validate(i)

        
    def train(self, epoch):
        widgets = [f'Train epoch {epoch} :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, max_value=int((self.train_samples // self.params.batch_size) + 20)).start()
        total_loss = 0

        for i, (images, labels) in pbar(enumerate(self.train_dataset)):
            loss = self.train_step(images, labels)
            total_loss += loss
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('train_step_loss', loss, step=self.checkpoint.train_steps)
            self.checkpoint.train_steps.assign_add(1)
        
        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_batch_loss', total_loss, step=epoch)
        
        self.checkpoint.epoch.assign_add(1)
        if int(self.checkpoint.epoch) % 5 == 0:
            save_path = self.ckptmanager.save()
            print('\nTrain Loss over epoch {}: {}'.format(epoch, total_loss))
            print(f'Saved Checkpoint for step {self.checkpoint.epoch.numpy()} : {save_path}\n')

            
    def validate(self, epoch):
        widgets = [f'Valid epoch {epoch} :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, max_value=int(self.valid_samples // self.params.batch_size) + 50).start()
        total_loss = 0

        for i, (images, labels) in pbar(enumerate(self.valid_dataset)):
            loss = self.valid_step(images, labels)
            total_loss += loss
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('valid_step_loss', loss, step=self.checkpoint.valid_steps)
            self.checkpoint.valid_steps.assign_add(1)
        print('\n')

        with self.train_summary_writer.as_default():
            tf.summary.scalar('valid_batch_loss', total_loss, step=epoch)
        
        if (epoch+1)%5 == 0:
            print('\nValidation Loss over epoch {}: {}\n'.format(epoch, total_loss)) 
    
        
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            embeddings = self.model(images)
            embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
            loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss
    
    
    def valid_step(self, images, labels):
        embeddings = self.model(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)
            
        return loss

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch',      default=1, type=int, help="Number epochs to train the model for")
    parser.add_argument('--params_dir', default='hyperparameters/batch_adaptive.json', help="Experiment directory containing params.json")
    parser.add_argument('--validate',   default='1', help="Is there an validation dataset available")
    parser.add_argument('--ckpt_dir',   default='/content/drive/Shareddrives/newpro112233/signal_machine/ckpt/', help="Directory containing the Checkpoints")
    parser.add_argument('--log_dir',    default='/content/drive/Shareddrives/newpro112233/signal_machine/log/', help="Directory containing the Logs")
    parser.add_argument('--emb_dir',    default='/content/drive/Shareddrives/newpro112233/signal_machine/emb1.npy', help="Directory containing the Checkpoints")
    parser.add_argument('--restore',    default='0', help="Restart the model from the previous Checkpoint")
    parser.add_argument('--threshold',  default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], type=str, help='num_mels')
    parser.add_argument('--faceNet',          default=True, type=bool)
    parser.add_argument('--Use_euclidean',    default=False, type=bool)

    # Models and denoising methods--------------------------
    parser.add_argument('--ML_method',   default='RandomForestClassifier', type=str)
    parser.add_argument('--use_DNN_A',   default=False, type=bool)
    parser.add_argument('--use_DNN_B',   default=False, type=bool)
    parser.add_argument('--use_CNN_A',   default=False, type=bool)
    parser.add_argument('--use_CNN_B',   default=False, type=bool)
    parser.add_argument('--use_CNN_C',   default=False, type=bool)
    parser.add_argument('--use_wavenet',       default=False, type=bool)
    parser.add_argument('--use_wavenet_head',  default=False, type=bool)
    parser.add_argument('--ensemble',          default=False, type=bool)
    parser.add_argument('--denoise', type=str, default=None, help='types of NN: DFK, Wavelet_denoise, SVD, savitzky_golay, None. DFK is our proposal.')
    parser.add_argument('--scaler',  type=str, default=None, help='handcrafted_features, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer')
    
    # Run case------------------------------------------------
    parser.add_argument('--case_0_6',  default=False,  type=bool)
    parser.add_argument('--case_1_7',  default=False,  type=bool)
    parser.add_argument('--case_2_8',  default=False,  type=bool)
    parser.add_argument('--case_3_9',  default=False,  type=bool)
    parser.add_argument('--case_4_10', default=False,  type=bool) # Turn on all cases before
    parser.add_argument('--case_5_11', default=False, type=bool)
    
    parser.add_argument('--case_12', default=False, type=bool) # turn on case_4_10
    parser.add_argument('--case_13', default=False,  type=bool)  # turn on case_5_11
    parser.add_argument('--case_14', default=False,  type=bool)  # turn on case 12 and case_4_11
    
    parser.add_argument('--PU_data_table_10',     default=True, type=bool)
    parser.add_argument('--PU_data_table_8',      default=False, type=bool)
    parser.add_argument('--MFPT_data',            default=False, type=bool)
    parser.add_argument('--data_normal',          default=False, type=bool)
    parser.add_argument('--data_12k',             default=False, type=bool)
    parser.add_argument('--data_48k',             default=False, type=bool)
    parser.add_argument('--multi_head',           default=False, type=bool)

    # Parameters---------------------------------------------
    parser.add_argument('--save',            type=str,   default='/content/drive/Shareddrives/newpro112233/signal_machine/', help='Position to save weights')
    parser.add_argument('--num_classes',     type=int,   default=512,          help='128 Number of classes in faceNet')
    parser.add_argument('--input_shape',     type=int,   default=255900,     help='127950 or 255900 in 5-fold or 250604 in the only training.')
    parser.add_argument('--batch_size',      type=int,   default=32,         help='Number of batch size for training')
    parser.add_argument('--test_rate',       type=float, default=0.2,        help='rate of split data for testing')
    parser.add_argument('--learning_rate',   type=float, default=0.001,      help='learning rate')

    parser.add_argument('--use_SNRdb',                type=bool,    default=False)
    parser.add_argument('--SNRdb',                    type=str,     default=[0, 5, 10, 15, 20, 25, 30],         help='intensity of noise')
    parser.add_argument('--num_mels',                 type=int,     default=80,          help='num_mels')
    parser.add_argument('--upsample_scales',          type=str,     default=[4, 8, 8],   help='num_mels')
    parser.add_argument('--model_names',              type=str,     default=['DNN', 'CNN_A', 'CNN_B', 'CNN_C', 'wavenet', 'wavelet_head'],   help='name of all NN models')
    parser.add_argument('--exponential_decay_steps',  type=int,     default=200000,      help='exponential_decay_steps')
    parser.add_argument('--exponential_decay_rate',   type=float,   default=0.5,         help='exponential_decay_rate')
    parser.add_argument('--beta_1',                   type=float,   default=0.9,         help='beta_1')
    parser.add_argument('--result_dir',               type=str,     default="./result/", help='exponential_decay_rate')
    parser.add_argument('--model_dir',                type=str,     default="/content/drive/Shareddrives/newpro112233/signal_machine/", help='direction to save model')
    parser.add_argument('--load_path',                type=str,     default=None,        help='path weight')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
    
if __name__ == '__main__':
    X_train_all, X_test, y_train_all, y_test = get_data(opt)
    opt = parse_opt()
    trainer = Trainer(opt, X_train_all, X_test, y_train_all, y_test)
    
    for i in range(opt.epoch):
        trainer.train(i)
