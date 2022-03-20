import tensorflow as tf
from tensorflow.python.keras import backend as K

# Accuracy for instance segmentation
def Instance_Segmentation_Accuracy(y_ture, y_pred):
    n = y_ture.shape[1]
    y_pred_round = tf.round(y_pred)
    sub_m = tf.abs(y_pred_round - y_ture)
    sub_m_add = tf.reduce_sum(sub_m, axis=-1)
    a = tf.zeros_like(sub_m_add)
    b = tf.ones_like(sub_m_add)
    sub_m_add_new = tf.where(sub_m_add < 0.5, x=a, y=b)
    sub_m_add_new_all = tf.reduce_sum(sub_m_add_new, axis=-1, keepdims=True)
    all = tf.constant([n], dtype=tf.float32)
    return K.cast(1.0 - sub_m_add_new_all / all, dtype=tf.float32)
