import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K

#Proposed loss function for instance segmentation
def Loss_Function(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    MySubtract = tf.sqrt(tf.square(tf.subtract(y_pred, y_true)))
    Loss = K.mean(1.0 / (1.0 + tf.exp(-150 * (MySubtract - 0.5))), axis=-1)

    return Loss