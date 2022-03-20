# Compress the data in the txt file to h5，最终上传的版本不需要改程序
import numpy as np
import random
import os
import Data_prep_util
from models import model_utils

myfilename = r'F:\ZH\DL_Pra\AllDatas\TXTDatas1\AllTogetherDatas(withoutAlldirection)\AllTogether\classify'
myfilenameSimilar = r'F:\ZH\DL_Pra\AllDatas\TXTDatas1\AllTogetherDatas(withoutAlldirection)\AllTogether\similar'
myfilenameBottom = r'F:\ZH\DL_Pra\AllDatas\TXTDatas1\AllTogetherDatas(withoutAlldirection)\AllTogether\classifybottom'

myfilenameh5 = r'F:\ZH\DL_Pra\AllDatas\TXTDatas1\AllTogetherDatas(withoutAlldirection)\AllTogether\zhh5shufflenewnormal'

FaceFeatures = model_utils.FaceFeatures
FaceNum = model_utils.FaceNum
PointFeatures = model_utils.PointFeatures
PointNum = model_utils.PointNum
Classnum = model_utils.Classnum

pointscloudfilenames0 = [ds for ds in os.listdir(myfilename)]
instancelabels0 = [ds1 for ds1 in os.listdir(myfilenameSimilar)]
bottomfacelabels0 = [ds1 for ds1 in os.listdir(myfilenameBottom)]
randomnumlist = list(range(len(pointscloudfilenames0)))
random.shuffle(randomnumlist)

pointscloudfilenames = []
instancelabels = []
bottomfacelabels = []
for i in randomnumlist:
    pointscloudfilenames.append(pointscloudfilenames0[i])
    instancelabels.append(instancelabels0[i])
    bottomfacelabels.append(bottomfacelabels0[i])


NameClass = ["traindata", "validationdata"]

for nc in NameClass:
    pointdatastrain = []
    pointdataslabeltrain = []
    pointdatasmarktrain = []
    pointdatasidtrain = []
    PartsName = []
    pointdatasPoF = []
    instanceSimilarity = []
    NumofInstances = []
    pointdatasbottomface = []

    save_path_train = os.path.join(myfilenameh5, nc + ".h5")
    if (len(pointscloudfilenames) == len(instancelabels)) & (len(bottomfacelabels) == len(instancelabels)):
        rate = 0.8
        if nc == "traindata":
            start = 0
            end = int(len(pointscloudfilenames) * rate)
        else:
            start = int(len(pointscloudfilenames) * rate)
            end = len(pointscloudfilenames)
        for i in range(start, end):
            pname = pointscloudfilenames[i]
            iname = instancelabels[i]
            bname = bottomfacelabels[i]
            if (pname[5:-4] == iname[8:-4]):#PartK
            #if True:
                pointdatastraininonepart = []
                instancelabelsforonepart = []
                cur_point_path = os.path.join(myfilename, pointscloudfilenames[i])
                cur_instancelabel_path = os.path.join(myfilenameSimilar, instancelabels[i])
                cur_bottom_path = os.path.join(myfilenameBottom, bottomfacelabels[i])
                if (os.path.isfile(cur_point_path)) & (
                        os.path.isfile(cur_instancelabel_path) & (os.path.isfile(cur_bottom_path))):
                    with open(cur_bottom_path) as f2:
                        lines = f2.readlines()
                        for line in lines:
                            data = list(map(str, (line.strip().split(' '))))
                            bottomfaceid = data
                    with open(cur_point_path) as f:
                        lines = f.readlines()
                        for line in lines:
                            data = list(map(str, (line.strip().split(' '))))
                            if data[0] != '':
                                pointdatastraininonepart.append(data)
                        pointdatastraininonepart = Data_prep_util.select_points(pointdatastraininonepart)
                        trainpointcloud = []
                        n = len(pointdatastraininonepart)
                        normal_pointsdata_zh = Data_prep_util.PointCloudNormal(Data_prep_util.getdatas(pointdatastraininonepart))
                        if (pointdatastraininonepart.shape[1] > 4):
                            trainpointcloud = Data_prep_util.add_other_data_withoutmark(pointdatastraininonepart,
                                                                                 normal_pointsdata_zh)
                        else:
                            trainpointcloud = normal_pointsdata_zh
                        pointslabel = Data_prep_util.getlabel(pointdatastraininonepart)
                        pointsmark = Data_prep_util.getmark(pointdatastraininonepart)
                        pointsid = Data_prep_util.getid(pointdatastraininonepart)
                        trainpointlabel = Data_prep_util.one_hot(pointslabel, Classnum)
                        ID = Data_prep_util.GetAllFacesID(pointsid)
                        PointsonFaces, FaceLabels, FaceMarks, FaceIds, FaceBottomLabels = Data_prep_util.GetPointsFromFaces_Label_Mark_Shuffle(
                            ID,
                            trainpointcloud,
                            trainpointlabel,
                            pointsmark,
                            pointsid,
                            pointdatastraininonepart,
                            bottomfaceid)
                    GlobalFaceIDs = FaceIds.copy()
                    if (len(ID) <= FaceNum):
                        PartsName.append(pname[5:-4])  # PartK
                        with open(cur_instancelabel_path) as f1:
                            instancelines = f1.readlines()
                            for instanceline in instancelines:
                                datainstance = list(map(str, (instanceline.strip().split(' '))))
                                instancelabelsforonepart.append(datainstance)
                        if (len(ID) < FaceNum):
                            addnum = FaceNum - len(ID)
                            for j in range(addnum):
                                F0 = []
                                for k in range(len(FaceLabels[0]) - 1):
                                    F0.append(0.0)
                                F0.append(1.0)
                                FaceLabels.append(F0)
                                FaceMarks.append('0.0')
                                GlobalFaceIDs.append('0.0')
                                FaceBottomLabels.append([0])
                                F1 = []
                                for k in range(len(PointsonFaces[0])):
                                    F2 = []
                                    for k in range(len(PointsonFaces[0][0])):
                                        F2.append(0.0)
                                    F1.append(F2)
                                PointsonFaces.append(F1)
                        SimilarityMatrix = np.eye(FaceNum)
                        allindexes = []
                        for Facesinoneinstance in instancelabelsforonepart:
                            if len(Facesinoneinstance) > 1:
                                indexes = []
                                for Faceinoneinstance in Facesinoneinstance:
                                    indexes.append(GlobalFaceIDs.index(Faceinoneinstance))
                                for j in range(len(indexes)):
                                    for k in range(j, len(indexes)):
                                        SimilarityMatrix[indexes[j]][indexes[k]] = 1
                                        SimilarityMatrix[indexes[k]][indexes[j]] = 1
                        zeroid = []
                        for j in range(len(GlobalFaceIDs)):
                            if GlobalFaceIDs[j] == '0.0':
                                zeroid.append(j)
                        for j in range(len(zeroid)):
                            for k in range(j, len(zeroid)):
                                SimilarityMatrix[zeroid[j]][zeroid[k]] = 1
                                SimilarityMatrix[zeroid[k]][zeroid[j]] = 1
                        NumofInstance = len(instancelabelsforonepart)
                        pointdataslabeltrain.append(FaceLabels)
                        pointdatasmarktrain.append(FaceMarks)
                        pointdatasidtrain.append(GlobalFaceIDs)
                        pointdatasPoF.append(PointsonFaces)
                        instanceSimilarity.append(SimilarityMatrix)
                        pointdatasbottomface.append(FaceBottomLabels)
                        NumofInstances.append(NumofInstance)
                        print(nc + " " + "num" + str(i) + " " + "file")
                    else:
                        print(nc + " " + "num" + str(i) + " " + "file : there are more than " + str(
                            FaceNum) + " faces in this part")
                else:
                    print("wrong")
                    print(str(i))
            else:
                print("wong")
                print(str(i))
        if (len(pointdataslabeltrain) != 0):
            print(len(pointdataslabeltrain))
            pointdatastrainzh = np.array(pointdatastrain, dtype=np.float64)
            pointdataslabeltrainzh = np.array(pointdataslabeltrain, dtype=np.float64)
            pointdatasmarktrainzh = np.array(pointdatasmarktrain, dtype=np.float64)
            pointdatasidtrainzh = np.array(pointdatasidtrain, dtype=np.float64)
            pointdatasPoFzh = np.array(pointdatasPoF, dtype=np.float64)
            PartsNamezh = np.array(PartsName, dtype=np.float64)
            instanceSimilarityzh = np.array(instanceSimilarity, dtype=np.float64)
            NumofInstanceszh = np.array(NumofInstances, dtype=np.float64)
            pointdatasbottomfacezh = np.array(pointdatasbottomface, dtype=np.float64)
            Data_prep_util.save_h5_mark_name(save_path_train, pointdatasPoFzh, pointdataslabeltrainzh, pointdatasmarktrainzh,
                                      PartsNamezh, pointdatasidtrainzh, instanceSimilarityzh, NumofInstanceszh,
                                      pointdatasbottomface)
    else:
        print("AllWrong")
        break