######################################################
# Original implementation by KinWaiCheuk: https://github.com/KinWaiCheuk/Triplet-net-keras
######################################################

from preprocessing.utils import to_one_hot
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.models import Model
from triplet import generate_triplet, triplet_loss
from tensorflow.keras.layers import concatenate, Lambda, Embedding
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import os
import argparse

def train(opt, x_train, y_train, x_test, y_test, network):
    print("\n Training with Triplet Loss....")

    outdir = opt.outdir + "/triplet_loss/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    model_input = Input(shape=(opt.input_shape, 1))
    softmax, pre_logits = network(opt, model_input)
    shared_model = tf.keras.models.Model(inputs=[model_input], outputs=[softmax, pre_logits])
   
    X_train, Y_train = generate_triplet(x_train, y_train)  #(anchors, positive, negative)
    X_test, Y_test = generate_triplet(x_test, y_test)
  
    anchor_input = Input((opt.input_shape, 1,), name='anchor_input')
    positive_input = Input((opt.input_shape, 1,), name='positive_input')
    negative_input = Input((opt.input_shape, 1,), name='negative_input')

    soft_anchor, pre_logits_anchor = shared_model([anchor_input])
    soft_pos, pre_logits_pos = shared_model([positive_input])
    soft_neg, pre_logits_neg = shared_model([negative_input])

    merged_pre = concatenate([pre_logits_anchor, pre_logits_pos, pre_logits_neg], axis=-1, name='merged_pre')
    merged_soft = concatenate([soft_anchor, soft_pos, soft_neg], axis=-1, name='merged_soft')
    
    loss_weights = [1, 0.01]
   
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=[merged_soft, merged_pre])
    model.load_weights(outdir + "triplet_loss_model.h5")
    model.compile(loss=["categorical_crossentropy", triplet_loss],
                  optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"], loss_weights=loss_weights)
    # https://keras.io/api/losses/
    
    anchor = X_train[:, 0, :].reshape(-1, opt.input_shape, 1)
    positive = X_train[:, 1, :].reshape(-1, opt.input_shape, 1)
    negative = X_train[:, 2, :].reshape(-1, opt.input_shape, 1)

    y_anchor = to_one_hot(Y_train[:, 0])
    y_positive = to_one_hot(Y_train[:, 1])
    y_negative = to_one_hot(Y_train[:, 2])

    target = np.concatenate((y_anchor, y_positive, y_negative), -1)
    model.load_weights(outdir + "triplet_loss_model.h5")
    model.fit(x=[anchor, positive, negative], y=[target, target],
              batch_size=opt.batch_size, epochs=opt.epoch, callbacks=[TensorBoard(log_dir=outdir)], validation_data=(X_test, Y_test))
    model.save(outdir + "triplet_loss_model.h5")

    # Embedding-------------------------------------------------------------------------------
    model = Model(inputs=[anchor_input], outputs=[soft_anchor, pre_logits_anchor])
    model.load_weights(outdir + "triplet_loss_model.h5")

    _, X_train_embed = model.predict([x_train])
    _, X_test_embed = model.predict([x_test])
    return X_train_embed, X_test_embed

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--outdir',   default='/content/drive/Shareddrives/newpro112233/signal_machine/runs/', help="Directory containing the Checkpoints")
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
    
    parser.add_argument('--PU_data_table_10',            default=True, type=bool)
    parser.add_argument('--PU_data_table_10_case_0',     default=False, type=bool)
    parser.add_argument('--PU_data_table_10_case_1',     default=True, type=bool)
    parser.add_argument('--PU_data_table_8',      default=False, type=bool)
    parser.add_argument('--MFPT_data',            default=False, type=bool)
    parser.add_argument('--data_normal',          default=False, type=bool)
    parser.add_argument('--data_12k',             default=False, type=bool)
    parser.add_argument('--data_48k',             default=False, type=bool)
    parser.add_argument('--multi_head',           default=False, type=bool)

    # Parameters---------------------------------------------
    parser.add_argument('--epoch',              type=int,   default=200, help="Number epochs to train the model for")
    parser.add_argument('--save',               type=str,   default='/content/drive/Shareddrives/newpro112233/signal_machine/', help='Position to save weights')
    parser.add_argument('--num_classes',        type=int,   default=3,          help='3 Number of classes in faceNet')
    parser.add_argument('--embedding_size',     type=int,   default=512,        help='128 Number of embedding in faceNet')
    parser.add_argument('--input_shape',        type=int,   default=255900,     help='127950 or 255900 in 5-fold or 250604 in the only training.')
    parser.add_argument('--batch_size',         type=int,   default=32,         help='Number of batch size for training')
    parser.add_argument('--test_rate',          type=float, default=0.2,        help='rate of split data for testing')
    parser.add_argument('--learning_rate',      type=float, default=0.001,      help='learning rate')

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
