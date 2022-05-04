# triplet loss
import tensorflow.keras.backend as K
from itertools import permutations
import random
import tensorflow as tf
import numpy as np
from train import parse_opt
opt = parse_opt()


def generate_triplet(x, y,  ap_pairs=8, an_pairs=8):
    data_xy = tuple([x, y])

    trainsize = 1

    triplet_train_pairs = []
    y_triplet_pairs = []
    #triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):
        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        A_P_pairs = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)  # Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        #Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            y_Anchor = data_xy[1][ap[0]]

            Positive = data_xy[0][ap[1]]
            y_Pos = data_xy[1][ap[1]]

            for n in Neg_idx:
                Negative = data_xy[0][n]
                y_Neg = data_xy[1][n]
                
                triplet_train_pairs.append([Anchor, Positive, Negative])
                y_triplet_pairs.append([y_Anchor, y_Pos, y_Neg])
                # test

    return np.array(triplet_train_pairs), np.array(y_triplet_pairs)

# def generate_triplet(x, y):
#     label = []
#     data = []
#     for i in sorted(set(y)):
#       anchor_idx = np.where(y==i)[0]
#       negative_idx = np.where(y!=i)[0]
#       # print(negative_idx, '\n')
#       if len(anchor_idx) > len(negative_idx):
#         negative_full_idx = random.sample(negative_idx.tolist(), k=len(negative_idx))*np.ceil(len(anchor_idx)/len(negative_idx)).astype(np.int32)
#         negative_idx = negative_full_idx[:len(anchor_idx)]
#       else:
#         negative_idx = negative_idx[:len(anchor_idx)]

#       # label pair-------------------------------
#       anchor_label = y[anchor_idx].reshape(-1, 1)
#       negative_label = y[negative_idx].reshape(-1, 1)

#       each_label = np.concatenate((anchor_label, negative_label), axis=1)
#       if label ==[]:
#         label = each_label
#       else:
#         label = np.concatenate((label, each_label))

#       # data pair--------------------------
#       anchor_data = x[anchor_idx]
#       negative_data = x[negative_idx]
#       each_data = np.concatenate((anchor_data, negative_data), axis=1)
#       if data ==[]:
#         data = each_data
#       else:
#         data = np.concatenate((data, each_data))
#     return data, label
    
def new_triplet_loss(y_true, y_pred, alpha=0.4, lambda_=opt.lambda_):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    total_lenght = y_pred.shape.as_list()[-1]

    anchor   = y_pred[:, 0: opt.embedding_size]
    # anchor   = tf.math.l2_normalize(anchor, axis=1, epsilon=1e-10)

    anchor_extract   = y_pred[:, opt.embedding_size: opt.embedding_size*2]
    # anchor_extract   = tf.math.l2_normalize(anchor_extract, axis=1, epsilon=1e-10)

    positive = y_pred[:, opt.embedding_size*2: opt.embedding_size*3]
    # positive = tf.math.l2_normalize(positive, axis=1, epsilon=1e-10)

    positive_extract = y_pred[:, opt.embedding_size*3: opt.embedding_size*4]
    # positive_extract = tf.math.l2_normalize(positive_extract, axis=1, epsilon=1e-10)

    negative = y_pred[:, opt.embedding_size*4: opt.embedding_size*5]
    # negative = tf.math.l2_normalize(negative, axis=1, epsilon=1e-10)

    negative_extract = y_pred[:, opt.embedding_size*5: opt.embedding_size*6]
    # negative_extract = tf.math.l2_normalize(negative_extract, axis=1, epsilon=1e-10)

    y_center = y_pred[:, opt.embedding_size*6: opt.embedding_size*7]
    y_center = tf.math.l2_normalize(y_center, axis=1, epsilon=1e-10)

    y_center_extract = y_pred[:, opt.embedding_size*7:]
    y_center_extract = tf.math.l2_normalize(y_center_extract, axis=1, epsilon=1e-10)
    

    pos_extract_dist  = K.sum(K.square(anchor_extract - positive_extract), axis=1)
    neg_extract_dist  = K.sum(K.square(anchor_extract - negative_extract), axis=1)
    out_extract_l2    = K.sum(K.square(anchor_extract - y_center_extract), axis=1) + pos_extract_dist
    loss_extract  = K.maximum(lambda_*out_extract_l2 + alpha - neg_extract_dist, 0.0)

    # mean ---------------------------------
    mean_anchor     = tf.expand_dims(tf.math.reduce_mean(anchor, axis=1), axis=1)
    multiple_anchor = tf.constant([1, anchor.shape.as_list()[1]])
    mean_anchor     = tf.tile(mean_anchor, multiple_anchor)

    mean_positive     = tf.expand_dims(tf.math.reduce_mean(positive, axis=1), axis=1)
    multiple_positive = tf.constant([1, positive.shape.as_list()[1]])
    mean_positive     = tf.tile(mean_positive, multiple_positive)

    mean_negative     = tf.expand_dims(tf.math.reduce_mean(negative, axis=1), axis=1)
    multiple_negative = tf.constant([1, negative.shape.as_list()[1]])
    mean_negative     = tf.tile(mean_negative, multiple_negative)

    # variance ------------------------------
    variance_anchor   = K.sum(K.square(anchor - mean_anchor), axis=1)/tf.cast(anchor.shape.as_list()[1], tf.float32)
    variance_positive = K.sum(K.square(positive - mean_positive), axis=1)/tf.cast(positive.shape.as_list()[1], tf.float32)
    variance_negative = K.sum(K.square(negative - mean_negative), axis=1)/tf.cast(negative.shape.as_list()[1], tf.float32)

    # distance between the anchor and the positive
    pos_dist          = K.sum(K.square(anchor - positive), axis=1)
    mean_pos_dist     = K.sum(K.square(mean_anchor - mean_positive))

    # distance between the anchor and the negative
    neg_dist          = K.sum(K.square(anchor - negative), axis=1)
    mean_neg_dist     = K.sum(K.square(mean_anchor - mean_negative))

    # compute loss
    out_l2     = K.sum(K.square(anchor - y_center), axis=1) + pos_dist
    loss       = K.maximum(lambda_*out_l2 + alpha - neg_dist, 0.0)

    mean_loss  = K.maximum(alpha - mean_neg_dist, 0.0)

    return loss + 0.3*loss_extract

def magnitudes(x):
  '''
  https://www.omnicalculator.com/math/angle-between-two-vectors
  '''
  return tf.math.sqrt(tf.math.reduce_sum(x**2))

def product(x, y):
  return tf.math.reduce_sum(x*y)

# def new_triplet_loss(y_true, y_pred):
#     """
#     Implementation of the triplet loss function
#     Arguments:
#     y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
#     y_pred -- python list containing three objects:
#             anchor -- the encodings for the anchor data
#             positive -- the encodings for the positive data (similar to anchor)
#             negative -- the encodings for the negative data (different from anchor)
#     Returns:
#     loss -- real number, value of the loss
#     """
#     alpha = (2*np.pi)/float(opt.num_classes)
#     total_lenght = y_pred.shape.as_list()[-1]

#     anchor   = y_pred[:, 0: opt.embedding_size]
#     magnitudes_anchor = magnitudes(anchor)

#     anchor_extract   = y_pred[:, opt.embedding_size: opt.embedding_size*2]
#     magnitudes_anchor_extract = magnitudes(anchor_extract)

#     negative = y_pred[:, opt.embedding_size*2: opt.embedding_size*3]
#     magnitudes_negative = magnitudes(negative)

#     negative_extract = y_pred[:, opt.embedding_size*3: opt.embedding_size*4]
#     magnitudes_negative_extract = magnitudes(negative_extract)


#     y_center = y_pred[:, opt.embedding_size*4: opt.embedding_size*5]
#     magnitudes_y_center = magnitudes(y_center)

#     y_center_extract = y_pred[:, opt.embedding_size*5:]
#     magnitudes_y_center_extract = magnitudes(y_center_extract)
    

#     # negative angle----------------------------------------
#     neg_dist          = tf.math.acos(product(anchor, negative) / (magnitudes_anchor*magnitudes_negative))
#     neg_extract_dist  = tf.math.acos(product(anchor_extract, negative_extract) / (magnitudes_anchor_extract*magnitudes_negative_extract))

#     # anchor angle------------------------------------------
#     out_l2            = tf.math.acos(product(anchor, y_center) / (magnitudes_anchor*magnitudes_y_center))
#     out_extract_l2    = tf.math.acos(product(anchor_extract, y_center_extract) / (magnitudes_anchor_extract*magnitudes_y_center_extract))
    
#     loss           = K.maximum(out_l2 + alpha - neg_dist, 0.0)
#     loss_extract   = K.maximum(out_extract_l2 + alpha - neg_extract_dist, 0.0)

#     return 0.5*loss + loss_extract

# def triplet_loss(y_true, y_pred, alpha=0.4, lambda_=opt.lambda_):
#     """
#     Implementation of the triplet loss function
#     Arguments:
#     y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
#     y_pred -- python list containing three objects:
#             anchor -- the encodings for the anchor data
#             positive -- the encodings for the positive data (similar to anchor)
#             negative -- the encodings for the negative data (different from anchor)
#     Returns:
#     loss -- real number, value of the loss
#     """
#     total_lenght = y_pred.shape.as_list()[-1]

#     anchor   = y_pred[:, 0:int(total_lenght * 1 / 3)]
#     anchor   = tf.math.l2_normalize(anchor, axis=1, epsilon=1e-10)
#     positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
#     positive = tf.math.l2_normalize(positive, axis=1, epsilon=1e-10)
#     negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]
#     negative = tf.math.l2_normalize(negative, axis=1, epsilon=1e-10)

#     # distance between the anchor and the positive
#     pos_dist          = K.sum(K.square(anchor - positive), axis=1)

#     # distance between the anchor and the negative
#     neg_dist          = K.sum(K.square(anchor - negative), axis=1)

#     # compute loss
#     basic_loss = pos_dist - neg_dist + alpha
#     loss       = K.maximum(basic_loss, 0.0)

#     return loss


def triplet_center_loss(y_true, y_pred, n_classes= 3, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    shape = y_pred.shape[1]//2
    pre_logits, center = y_pred[:, :shape], y_pred[:, shape:]
    y_pred = K.sum(K.square(pre_logits - center), axis=1)

    # repeat y_true for n_classes and == np.arange(n_classes)
    # repeat also y_pred and apply mask
    # obtain min for each column min vector for each class

    classes = tf.range(0, n_classes,dtype=tf.float32)
    y_pred_r = tf.reshape(y_pred, (tf.shape(y_pred)[0], 1))
    y_pred_r = tf.keras.backend.repeat(y_pred_r, n_classes)

    y_true_r = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
    y_true_r = tf.keras.backend.repeat(y_true_r, n_classes)

    mask = tf.equal(y_true_r[:, :, 0], classes)

    #mask2 = tf.ones((tf.shape(y_true_r)[0], tf.shape(y_true_r)[1]))  # todo inf

    # use tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32) #+ (mask2 * tf.cast(tf.logical_not(mask), tf.float32))*tf.constant(float(2**10))
    masked = tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    minimums = tf.math.reduce_min(masked, axis=1)

    loss = K.max(y_pred - minimums +alpha ,0)

    # obtain a mask for each pred


    return loss
