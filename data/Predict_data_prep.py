#最终上传的版本不需要该程序
import os
import Data_prep_util
import numpy as np


myfilename=r'F:\ZH\DL_Pra\AllDatas\PaperDatas\PartDraw\ForDemo\pointcloud'#预测txt所在文件夹

myfilenameh5=r'F:\ZH\DL_Pra\AllDatas\PaperDatas\PartDraw\ForDemo\zhh5'#h5存放位置

# myfilename=r'E:\ZH\WORK\FSN\demodatatxt'#预测txt所在文件夹
# myfilenameh5=r'E:\ZH\WORK\FSN\demodatatxt\h5'#h5存放位置

# EmbeddingFeatures=32

FaceFeatures=32 #面的信息维度
FaceNum=64 #共取得48个面
PointFeatures=6 #点的信息维度
PointNum=32 #每个面5个点
classnum=4
EmbeddingFeatures=32

#假设FaceNum个面

pointscloudfilenames=[ds for ds in os.listdir(myfilename)]

pointdataspredict=[]
allFaceIds = []
allFaceIdszh = []
PartsName=[]
PartsNamezh=[]
pointdatasPoF=[]#以面上的点进行储存

pointdatasbottomface = []

dirnames=os.path.split(myfilename)#分割路径名，保留最后一个名称
dirname=dirnames[1]
save_path=os.path.join(myfilenameh5, "predict"+dirname+"Shuffle.h5")


for i in range(int(len(pointscloudfilenames))):
    pname=pointscloudfilenames[i]
    #PartsName.append(pname[4:-4])#Part
    pointdataspredictinonepart=[]
    cur_point_path = os.path.join(myfilename,pointscloudfilenames[i])
    if os.path.isfile(cur_point_path):
        with open(cur_point_path) as f:
            lines = f.readlines()
            for line in lines:
                data = list(map(str, (line.strip().split(' '))))
                pointdataspredictinonepart.append(data)
            pointdataspredictinonepart=Data_prep_util.select_points(pointdataspredictinonepart)
            predictpointcloud = []
            n = len(pointdataspredictinonepart)
            normal_pointsdata_zh = Data_prep_util.PointCloudNormal(Data_prep_util.getdatas(pointdataspredictinonepart))  # zh_method
            all_points_normal =  Data_prep_util.getallnormal(pointdataspredictinonepart)
            normal_pointsdata_zh = np.array(normal_pointsdata_zh)
            all_points_normal = np.array(all_points_normal)
            predictpointcloud = np.hstack((normal_pointsdata_zh, all_points_normal))

            pointsid =  Data_prep_util.getallFaceid(pointdataspredictinonepart)
            ID=Data_prep_util.GetAllFacesID(pointsid)
            allFaceIds.append(ID)
            FinalPointsOnFace, FinalFaceIDs = Data_prep_util.GetPointsFromFaces_Label_For_Pre(ID, predictpointcloud ,pointsid, pointdataspredictinonepart)

            if (len(ID) <= FaceNum):
                PartsName.append(pname[4:-4])  # Part
                if (len(ID) < FaceNum):
                    addnum = FaceNum - len(ID)
                    for j in range(addnum):
                        F1 = []
                        FinalFaceIDs.append(0.0)
                        for j in range(len(FinalPointsOnFace[0])):
                            F2 = []
                            for k in range(len(FinalPointsOnFace[0][0])):
                                F2.append(0.0)
                            F1.append(F2)
                        FinalPointsOnFace.append(F1)

            pointdatasPoF.append(FinalPointsOnFace)
            allFaceIdszh.append(FinalFaceIDs)
            PartsNamezh.append(PartsName)
            print("predict num"+str(i)+" "+"files")

if(len(pointdatasPoF)!=0):
    pointdatasPoFzh=np.array(pointdatasPoF,dtype=np.float64)
    PartsNamezh = np.array(PartsName,dtype=np.float64)
    allFaceIdszh = np.array(allFaceIdszh,dtype=np.float64)
    Data_prep_util.save_h5_for_pre(save_path, pointdatasPoFzh, PartsNamezh, allFaceIdszh)
