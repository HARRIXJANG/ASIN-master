import os
import Data_prep_util
import numpy as np

##############################################slot-normal########################################################
myfilename=r'F:\ZH\DL_Pra\AllDatas\TestDatas\TestDatas\classify'#预测txt所在文件夹
myfilenameSimilar=r'F:\ZH\DL_Pra\AllDatas\TestDatas\TestDatas\similar'#相似矩阵
myfilenameBottom=r'F:\ZH\DL_Pra\AllDatas\TestDatas\TestDatas\classifybottom'#底面

myfilenameh5=r'F:\ZH\DL_Pra\AllDatas\TestDatas\TestDatas\zhh5'#h5存放位置


FaceFeatures=32 #面的信息维度
FaceNum=64 #共取得48个面
PointFeatures=6 #点的信息维度
PointNum=32 #每个面5个点
classnum=4
EmbeddingFeatures=32

pointscloudfilenames=[ds for ds in os.listdir(myfilename)]
instancelabels=[ds1 for ds1 in os.listdir(myfilenameSimilar)]
bottomfacelabels=[ds1 for ds1 in os.listdir(myfilenameBottom)] #bottomface

pointdataspredict=[]
pointdataslabelpredict=[]
pointdatasmarkpredict=[]
pointdatasidpredict=[]
PartsName=[]

pointdatasPoF=[]#以面上的点进行储存
instanceSimilarity = []  # 各个面的相似矩阵
NumofInstances=[]

pointdatasbottomface = []

dirnames=os.path.split(myfilename)#分割路径名，保留最后一个名称
dirname=dirnames[1]
save_path=os.path.join(myfilenameh5,"predict"+dirname+"Shuffle.h5")
for i in range(int(len(pointscloudfilenames))):
    pname=pointscloudfilenames[i]
    iname = instancelabels[i]  # instance
    #PartsName.append(pname[4:-4])#Part
    pointdataspredictinonepart=[]
    instancelabelsforonepart=[]
    cur_point_path = os.path.join(myfilename,pointscloudfilenames[i])
    cur_instancelabel_path = os.path.join(myfilenameSimilar, instancelabels[i])
    cur_bottom_path = os.path.join(myfilenameBottom, bottomfacelabels[i])
    if os.path.isfile(cur_point_path):
        with open(cur_bottom_path) as f2:
            lines = f2.readlines()
            for line in lines:
                data = list(map(str, (line.strip().split(' '))))
                bottomfaceid = data
        with open(cur_point_path) as f:
            lines = f.readlines()
            for line in lines:
                data = list(map(str, (line.strip().split(' '))))
                pointdataspredictinonepart.append(data)
            pointdataspredictinonepart=Data_prep_util.select_points(pointdataspredictinonepart)
            predictpointcloud = []
            n = len(pointdataspredictinonepart)
            normal_pointsdata_zh = Data_prep_util.PointCloudNormal(Data_prep_util.getdatas(pointdataspredictinonepart))  # zh_method
            if(pointdataspredictinonepart.shape[1]>4):
                predictpointcloud = Data_prep_util.add_other_data_withoutmark(pointdataspredictinonepart,normal_pointsdata_zh)
            else:
                predictpointcloud=normal_pointsdata_zh
            pointslabel = Data_prep_util.getlabel(pointdataspredictinonepart)
            pointsmark = Data_prep_util.getmark(pointdataspredictinonepart)
            pointsid = Data_prep_util.getid(pointdataspredictinonepart)
            predictpointlabel = Data_prep_util.one_hot(pointslabel,classnum)
            ID=Data_prep_util.GetAllFacesID(pointsid)
            PointsonFaces,FaceLabels,FaceMarks,FaceIds,FaceBottomLabels=Data_prep_util.GetPointsFromFaces_Label_Mark_Shuffle(ID,
                                                                                     predictpointcloud,
                                                                                     predictpointlabel,
                                                                                     pointsmark,
                                                                                     pointsid,
                                                                                     pointdataspredictinonepart,
                                                                                     bottomfaceid)
        if (len(ID) <= FaceNum):
            PartsName.append(pname[5:-4])  # PartK
            with open(cur_instancelabel_path) as f1:  # instance
                instancelines = f1.readlines()
                for instanceline in instancelines:
                    datainstance = list(map(str, (instanceline.strip().split(' '))))
                    instancelabelsforonepart.append(datainstance)
            if(len(ID)<FaceNum):
                addnum=FaceNum-len(ID)
                for j in range(addnum):
                    F0=[]
                    for j in range(len(FaceLabels[0])-1):
                        F0.append(0.0)
                    F0.append(1.0)
                    FaceLabels.append(F0)
                    FaceMarks.append('0.0')
                    FaceIds.append('0.0')
                    FaceBottomLabels.append([0])
                    F1=[]
                    for j in range(len(PointsonFaces[0])):
                        F2=[]
                        for k in range(len(PointsonFaces[0][0])):
                            F2.append(0.0)
                        F1.append(F2)
                    PointsonFaces.append(F1)
            SimilarityMatrix = np.eye(FaceNum)  # instance
            allindexes = []
            for Facesinoneinstance in instancelabelsforonepart:
                if len(Facesinoneinstance) > 1:
                    indexes = []
                    for Faceinoneinstance in Facesinoneinstance:
                        indexes.append(FaceIds.index(Faceinoneinstance))
                    for j in range(len(indexes)):
                        for k in range(j, len(indexes)):
                            SimilarityMatrix[indexes[j]][indexes[k]] = 1
                            SimilarityMatrix[indexes[k]][indexes[j]] = 1
            zeroid = []
            for j in range(len(FaceIds)):
                if FaceIds[j] == '0.0':
                    zeroid.append(j)
            for j in range(len(zeroid)):
                for k in range(j, len(zeroid)):
                    SimilarityMatrix[zeroid[j]][zeroid[k]] = 1
                    SimilarityMatrix[zeroid[k]][zeroid[j]] = 1
            NumofInstance = len(instancelabelsforonepart)
            #pointdataspredict.append(predictpointcloud)
            pointdataslabelpredict.append(FaceLabels)
            pointdatasmarkpredict.append(FaceMarks)
            pointdatasidpredict.append(FaceIds)
            pointdatasPoF.append(PointsonFaces)
            NumofInstances.append(NumofInstance)
            instanceSimilarity.append(SimilarityMatrix)
            pointdatasbottomface.append(FaceBottomLabels)
            print("predict num"+str(i)+" "+"files")
        else:
            print("predict num" + str(i) + " " + "file : there are more than " + str(FaceNum) + " faces in this part")
if(len(pointdataslabelpredict)!=0):
#pointdataspredictzh = np.array(pointdataspredict, dtype=np.float64)
    pointdataslabelpredictzh = np.array(pointdataslabelpredict, dtype=np.float64)
    pointdatasmarkpredictzh = np.array(pointdatasmarkpredict,dtype=np.float64)
    pointdatasidpredictzh = np.array(pointdatasidpredict,dtype=np.float64)
    pointdatasPoFzh=np.array(pointdatasPoF,dtype=np.float64)
    PartsNamezh = np.array(PartsName,dtype=np.float64)
    instanceSimilarityzh=np.array(instanceSimilarity, dtype=np.float64)
    NumofInstanceszh=np.array(NumofInstances, dtype=np.float64)
    pointdatasbottomfacezh=np.array(pointdatasbottomface, dtype=np.float64)
    Data_prep_util.save_h5_mark_name(save_path, pointdatasPoFzh, pointdataslabelpredictzh, pointdatasmarkpredictzh,
                              PartsNamezh, pointdatasidpredictzh,instanceSimilarityzh,NumofInstanceszh,pointdatasbottomfacezh)
