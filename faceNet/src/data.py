import os
import numpy as np
# import pandas as pd
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
Create tf.data.Dataset from a directory of Images.
Preprocess Images : Resize, Flip, Normalize
'''


def preprocess_image(image, image_size):
    
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.random_flip_left_right(image)
    image /= 255.0
    return image


def parse_image_function(image_path, label, image_size):
      
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = preprocess_image(image, image_size)
    return image, label


def get_dataset(data, labels, params, phase='train'):

    AUTOTUNE   =  tf.data.experimental.AUTOTUNE
    dataset    =  tf.data.Dataset.from_tensor_slices((data, labels))
    dataset    =  dataset.map(lambda x, y: parse_image_function(x, y, params.image_size))
    dataset    =  dataset.batch(params.batch_size).prefetch(AUTOTUNE)
    
    return dataset, int(data.shape[0])


if __name__ == "__main__":
    print(get_dataset('../data'))
