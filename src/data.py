import os
import numpy as np
from preprocessing.utils import invert_one_hot
import tensorflow as tf

def get_dataset(data, labels, params, phase='train'):
    AUTOTUNE   =  tf.data.experimental.AUTOTUNE
    dataset    =  tf.data.Dataset.from_tensor_slices((data, labels))
    dataset    =  dataset.batch(params.batch_size).prefetch(AUTOTUNE)
    
    return dataset, int(data.shape[0])


