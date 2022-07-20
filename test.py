import os
import h5py
import numpy as np
from models import model

def load_h5_name_id(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    bflabel = f['bottomfacelabel'][:]
    similarmatrix = f['similarity'][:]
    numofinstances = f['numofinstances'][:]
    nameofpart = f['name'][:]
    id = f['id'][:]
    return (data, label, bflabel, similarmatrix, numofinstances,nameofpart, id)

# path to test data set
filenames_predict = [r'data\testdata.h5']

# path to the best model weights
model_save_weights_file = r'models\ASIN_best_weights.h5'

test_points_OnFace = None
test_labels = None
test_bottomlabels = None
test_similar_matrix = None
test_numofinstances = None
test_namesofpart = None
test_FacesId = None

for d in filenames_predict:
    cur_points_OnFace, cur_labels, cur_bottomfacelabel, cur_similar_matrix, cur_numofinstances, NamesofPart, FacesID = load_h5_name_id(d)
    cur_points_OnFace = cur_points_OnFace[:, :, :32, :]
    if test_labels is None or test_points_OnFace is None:
        test_labels = cur_labels
        test_points_OnFace = cur_points_OnFace
        test_bottomlabels = cur_bottomfacelabel
        test_similar_matrix = cur_similar_matrix
        test_numofinstances = cur_numofinstances
        test_namesofpart = NamesofPart
        test_FacesId = FacesID
    else:
        test_labels = np.vstack((test_labels, cur_labels))
        test_points_OnFace = np.vstack((test_points_OnFace, cur_points_OnFace))
        test_bottomlabels = np.vstack((test_bottomlabels. cur_bottomfacelabel))
        test_similar_matrix = np.vstack((test_similar_matrix, cur_similar_matrix))
        test_numofinstances = np.hstack((test_numofinstances, cur_numofinstances))
        test_namesofpart = np.hstack((test_numofinstances, cur_numofinstances))
        test_FacesId = np.hstack((test_numofinstances, cur_numofinstances), dtype=int)
predict_numofinstances = test_numofinstances.reshape((len(test_numofinstances), 1))
predict_namesofpart = test_namesofpart.reshape((len(test_namesofpart), 1))

loaded_model = model.ASIN_model()
loaded_model.summary()
loaded_model.load_weights(model_save_weights_file)
results = loaded_model.predict(test_points_OnFace)

r_Feature_Recogintion = np.round(results[0])
r_Bottom_Face_Recogintion = np.round(results[1])
r_Instances_Segment = np.round(results[2])

k_FR = 0
for i in range(r_Feature_Recogintion.shape[0]):
    for j in range(r_Feature_Recogintion.shape[1]):
        tt = np.array(r_Feature_Recogintion[i, j, :], dtype=np.int)
        tt2 = test_labels[i, j, :]
        if all(tt == tt2):
            k_FR  += 1
acc_Feature_Recogintion = k_FR / (r_Feature_Recogintion.shape[0]*r_Feature_Recogintion.shape[1])
print("Feature_Recognition: "+str(acc_Feature_Recogintion*100)+"%")

k_BFR = 0
for i in range(r_Bottom_Face_Recogintion.shape[0]):
    for j in range(r_Bottom_Face_Recogintion.shape[1]):
        tt = np.array(r_Bottom_Face_Recogintion[i, j, :], dtype=np.int)
        tt2 = test_bottomlabels[i, j, :]
        if all(tt == tt2):
            k_BFR += 1
acc_Feature_Recogintion_Bottom = k_BFR / (r_Bottom_Face_Recogintion.shape[0]*r_Bottom_Face_Recogintion.shape[1])
print("Bottom_Face_Recognition: "+str(acc_Feature_Recogintion_Bottom*100)+"%")

k_IS = 0
for i in range(r_Instances_Segment.shape[0]):
    for j in range(r_Instances_Segment.shape[1]):
        tt = np.array(r_Instances_Segment[i, j, :], dtype=np.int)
        tt2 = test_similar_matrix[i, j, :]
        if all(tt == tt2):
            k_IS += 1
acc_Instances_Segment = k_IS / (r_Instances_Segment.shape[0]*r_Instances_Segment.shape[1])
print("Instance_Segment: "+str(acc_Instances_Segment*100)+"%")
