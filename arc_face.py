import os
from functools import partial
import tensorflow as tf
import keras
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate
import keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda
from tensorflow.keras.models import Model
from angular_grad import AngularGrad
from src.arc_model import ArcFaceModel
from src.losses import SoftmaxLoss

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)
          
def train_ArcFaceModel(opt, x_train, y_train, x_test, y_test, i=100):
    print("\n Training with Triplet Loss....")

    outdir = opt.outdir + "/ArcFace/"
    if i==0:
      epoch = 50 # 30
    else:
      epoch = opt.epoch # 10

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        
    loss_fn = SoftmaxLoss()
    
    model = ArcFaceModel(opt=opt)
    # model.summary(line_length=80)
          
    model.compile(optimizer=AngularGrad(), loss=loss_fn)

    if opt.use_weight:
      if os.path.isdir(outdir + "ArcFace"):
        model.load_weights(outdir + "ArcFace")
        print(f'\n Load weight: {outdir}')
      else:
        print('\n No weight file.')
    # Fit data-------------------------------------------------
    model.fit(x=[x_train, y_train], y=y_train,
              batch_size=opt.batch_size, epochs=epoch, 
              # callbacks=[callback], 
              shuffle=True)
    tf.saved_model.save(model, outdir + 'ArcFace')
    
    model = ArcFaceModel(opt=opt, training=False)
    model.load_weights(outdir + 'ArcFace')
    
    X_train_embed = model.predict([x_train])
    X_test_embed = model.predict([x_test])

    from TSNE_plot import tsne_plot
    tsne_plot(outdir, 'ArcFace', X_train_embed, X_test_embed, y_train, y_test)
    return X_train_embed, X_test_embed, y_train, outdir
