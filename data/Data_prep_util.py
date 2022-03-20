#把txt数据压如h5,带有面ID和特征序号，最终上传的版本不需要改程序
import numpy as np
import random
import h5py

def save_h5_mark_name(h5_filename, data, label, mark ,name, id, InstanceSimilarity, NumofInstances, pointdatasbottomface, data_dtype='float', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.create_dataset(
        'mark', data=mark,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.create_dataset(
        'name', data=name,
         compression='gzip', compression_opts=1,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'id', data=id,
        compression='gzip', compression_opts=1,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'similarity', data=InstanceSimilarity,
        compression='gzip', compression_opts=1,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'numofinstances', data=NumofInstances,
        compression='gzip', compression_opts=1,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'bottomfacelabel', data=pointdatasbottomface,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_fout.close()

def save_h5_for_pre(h5_filename, data, name, id, data_dtype='float', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'name', data=name,
         compression='gzip', compression_opts=1,
        dtype=data_dtype)
    h5_fout.create_dataset(
        'id', data=id,
        compression='gzip', compression_opts=1,
        dtype=data_dtype)
    h5_fout.close()


def select_points(points):
    selected_points = np.array(points)
    return selected_points

def getnumlabel(onehot):
    return onehot.find('1')

def one_hot(label_names,classnum):
    numa=len(label_names)
    numb=classnum
    lns=np.zeros((numa,numb))
    for i in range(int(len(label_names))):
       lns[i][int(label_names[i])]=1
    lns=lns.tolist()
    return lns


def getalldatas(mydatas):
    datas=[]
    for mydata in mydatas:
        datas.append(float(mydata[0]))
        datas.append(float(mydata[1]))
        datas.append(float(mydata[2]))
    return datas

def getdatas(mydatas):
    datas = []
    for mydata in mydatas:
        data = []
        data.append(float(mydata[0]))
        data.append(float(mydata[1]))
        data.append(float(mydata[2]))
        datas.append(data)
    return datas

def PointCloudNormal(pointCloud):
    pointCloud = np.array(pointCloud)
    pointCloud_XYZ=pointCloud[:,0:3]
    x_min=np.min(pointCloud_XYZ[:,0])
    x_max=np.max(pointCloud_XYZ[:,0])
    y_min=np.min(pointCloud_XYZ[:,1])
    y_max=np.max(pointCloud_XYZ[:,1])
    z_min=np.min(pointCloud_XYZ[:,2])
    z_max=np.max(pointCloud_XYZ[:,2])

    x_range=x_max-x_min
    y_range=y_max-y_min
    z_range=z_max-z_min

    pointcloud_scale=min([x_range,y_range,z_range])

    pointCloud[:, 0] -= x_min
    pointCloud[:, 1] -= y_min
    pointCloud[:, 2] -= z_min

    pointCloud[:, 0:3]/=pointcloud_scale
    return pointCloud

def add_other_data_withoutmark(alldatas,orddatas):
    orddatas=np.array(orddatas, dtype=np.float64)
    numa=alldatas.shape[0]
    numb=alldatas.shape[1]-6
    newdatas=np.zeros((numa,numb))
    for i in range(numa):
        k = 0
        for j in range(3,alldatas.shape[1]-3):
            newdatas[i][k]=alldatas[i][j]
            k += 1
    newdatas=np.hstack((orddatas,newdatas))
    newdatas.tolist()
    return newdatas

def getlabel(datas):
    labels=[]
    for data in datas:
        labels.append(getnumlabel(data[-1]))
    return labels

def getmark(datas):
    marks=[]
    for data in datas:
        marks.append(data[-2])
    return marks

def getid(datas):
    marks=[]
    for data in datas:
        marks.append(data[-3])
    return marks

def getallFaceid(mydatas):
    datas=[]
    for mydata in mydatas:
        datas.append(mydata[-1])
    return datas



def getallnormal(mydatas):
    datas=[]
    for mydata in mydatas:
        data = []
        data.append(float(mydata[3]))
        data.append(float(mydata[4]))
        data.append(float(mydata[5]))
        datas.append(data)
    return datas

def GetAllFacesID(PointsId):
    ID=[]
    for i in range(len(PointsId)):
        if i==0:
            ID.append(PointsId[i])
        else:
            if PointsId[i] not in ID:
                ID.append(PointsId[i])
    return ID

def GetPointsFromFaces_Label_Mark_Shuffle(ID, predictpointcloud, predictpointlabel, pointsmark, pointsid,
                                  pointdataspredictinonepart,  bottomfaceid):
    IdLen = len(ID)
    PointsonFaces = []
    FaceMarks = []
    FaceLabels = []
    FaceBottomLabels = []
    FaceIDs = []

    randomIdlist = list(range(IdLen))
    random.shuffle(randomIdlist)

    for j in randomIdlist:
        myPointonFace = []
        for i in range(len(pointdataspredictinonepart)):
            if pointdataspredictinonepart[i][-3] == ID[j]:
                myPointonFace.append(predictpointcloud[i].tolist())
                FaceLabel = predictpointlabel[i]
                FaceLabel.append(0.0)
                FaceMark = pointsmark[i]
                FaceID = pointsid[i]
                if FaceID in bottomfaceid:
                    FaceBottomLabel = [1]
                else:
                    FaceBottomLabel = [0]
        PointsonFaces.append(myPointonFace)
        FaceMarks.append(FaceMark)
        FaceLabels.append(FaceLabel)
        FaceIDs.append(FaceID)
        FaceBottomLabels.append(FaceBottomLabel)

    return PointsonFaces, FaceLabels, FaceMarks, FaceIDs, FaceBottomLabels

def GetPointsFromFaces_Label_For_Pre(ID, predictpointcloud, pointsid, pointdataspredictinonepart):  # 按面排列所有面上的点
    IdLen = len(ID)
    PointsonFaces = []
    FaceIDs = []

    randomIdlist = list(range(IdLen))
    random.shuffle(randomIdlist)

    for j in randomIdlist:
        myPointonFace = []
        for i in range(len(pointdataspredictinonepart)):
            if pointdataspredictinonepart[i][-1] == ID[j]:
                myPointonFace.append(predictpointcloud[i].tolist())
                FaceID = pointsid[i]
        PointsonFaces.append(myPointonFace)
        FaceIDs.append(FaceID)

    return PointsonFaces, FaceIDs