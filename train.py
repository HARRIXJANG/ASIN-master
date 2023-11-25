import h5py
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras import backend as K
from models import model
from loss import Loss_Function
from acc import Instance_Segmentation_Accuracy

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    similarmatrix = f['similarity'][:]
    numofinstances = f['numofinstances'][:]
    bottomlabel = f['bottomfacelabel'][:]
    return (data, label, similarmatrix, numofinstances, bottomlabel)

# pathes to training data set and validation data set
paths = [r'data\traindata.h5', r'data\validationdata.h5']

# path to model weights
model_save_weights_file = 'models\ASIN_weights.h5'

# path to the best model weights
best_model_save_weithts_file = 'models\ASIN_best_weights.h5'

# path to the logging directory
logdir = '\logdir'

filenames_train = []
for d in paths:
    t = 'train'
    if t in d:
        filenames_train.append(d)

print(filenames_train)
train_points_OnFace = None
train_labels = None
train_similar_matrix = None
train_numofinstances = None
train_bottom_labels = None

for d in filenames_train:
    print(d)
    cur_points_OnFace, cur_labels, cur_similar_matrix, cur_numofinstances, cur_bottom_labels = load_h5(d)
    cur_points_OnFace = cur_points_OnFace[:, :, :32, :]
    if train_labels is None or train_points_OnFace is None:
        train_labels = cur_labels
        train_points_OnFace = cur_points_OnFace
        train_similar_matrix = cur_similar_matrix
        train_numofinstances = cur_numofinstances
        train_bottom_labels = cur_bottom_labels
    else:
        train_labels = np.vstack((train_labels, cur_labels))
        train_points_OnFace = np.vstack((train_points_OnFace, cur_points_OnFace))  # 按照矩阵的行进行拼接
        train_similar_matrix = np.vstack((train_similar_matrix, cur_similar_matrix))
        train_bottom_labels = np.vstack((train_bottom_labels, cur_bottom_labels))
        train_numofinstances = np.hstack((train_numofinstances, cur_numofinstances))
train_numofinstances = train_numofinstances.reshape((len(train_numofinstances), 1))
filenames_validation = []
for d in paths:
    t = 'validation'
    if t in d:
        filenames_validation.append(d)

validation_points_OnFace = None
validation_labels = None
validation_similar_matrix = None
validation_numofinstances = None
validation_bottom_labels = None

for d in filenames_validation:
    cur_points_OnFace, cur_labels, cur_similar_matrix, cur_numofinstances, cur_bottom_labels = load_h5(d)
    cur_points_OnFace = cur_points_OnFace[:, :, :32, :]
    if validation_labels is None or validation_points_OnFace is None:
        validation_labels = cur_labels
        validation_points_OnFace = cur_points_OnFace
        validation_similar_matrix = cur_similar_matrix
        validation_numofinstances = cur_numofinstances
        validation_bottom_labels = cur_bottom_labels
    else:
        validation_labels = np.vstack((validation_labels, cur_labels))
        validation_points_OnFace = np.vstack((validation_points_OnFace, cur_points_OnFace))
        validation_similar_matrix = np.vstack((validation_similar_matrix, cur_similar_matrix))
        validation_bottom_labels = np.vstack((validation_bottom_labels, cur_bottom_labels))
        validation_numofinstances = np.hstack((validation_numofinstances, cur_numofinstances))
validation_numofinstances = validation_numofinstances.reshape((len(validation_numofinstances), 1))

mymodel = model.ASIN_model()
mymodel.summary()

def scheduler(epoch):  # decrease learning rate
    if epoch < 120:
        K.set_value(mymodel.optimizer.lr, 0.001)
    else:
        if (epoch - 120) % 30 == 0 or epoch == 120:
            lr = K.get_value(mymodel.optimizer.lr)
            K.set_value(mymodel.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
    return K.get_value(mymodel.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)

optimizer_adam = optimizers.Adam(lr=0.001)
mymodel.compile(optimizer=optimizer_adam,
                loss=[losses.categorical_crossentropy,
                      losses.binary_crossentropy,
                      Loss_Function],
                loss_weights=[1, 1, 3],
                metrics=['accuracy', Instance_Segmentation_Accuracy],
                )

time_start=time.time()

with tf.device('/GPU:0'):
    tbcallbacktrain = TensorBoard(log_dir=logdir)
    checkpoint = ModelCheckpoint(best_model_save_weithts_file, monitor='val_loss', save_best_only=True, mode='auto',
                                 period=1)
    scoretrain = mymodel.fit(train_points_OnFace,
                             [train_labels, train_bottom_labels, train_similar_matrix],
                             batch_size=32, epochs=240, shuffle=True,
                             validation_data=(validation_points_OnFace, [validation_labels, validation_bottom_labels, validation_similar_matrix]),
                             callbacks=[tbcallbacktrain, reduce_lr, checkpoint])
    mymodel.save_weights(model_save_weights_file)
    time_end = time.time()
    print('totally cost', time_end - time_start)
