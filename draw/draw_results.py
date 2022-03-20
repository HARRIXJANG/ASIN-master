import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py

# path to the demo data
h5path = r'demodata.h5'

# path to the predict results
resultfile = r'Results.txt'

# Parts to be drawn, e.g. 'Part1', 'Part2', 'Part3',etc.
partname = 'Part2'

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    partname = f['name'][:]
    facename = f['id'][:]
    return data, partname,facename

def ReadTxt(path):
    partnames = []
    faceids =[]
    faceclasses = []
    bottomfacelabel = []
    instancefaces = []
    with open(path) as p:
        lines = p.readlines()
        for i in range(len(lines)):
            if i != 0 & i != len(lines)-1:
                data = list(map(str, (lines[i].strip(). split('\t'))))
                if i % 6 == 1:
                    partnum = data[0][4:]
                    if partnum != " ":
                        partnames.append(partnum)
                elif i % 6 == 2:
                    faceids.append(data)
                elif i % 6 == 3:
                    faceclasses.append(data)
                elif i % 6 == 4:
                    bottomfacelabel.append(data)
                elif i % 6 == 5:
                    instancefaces.append(data)
    return  partnames, faceids, faceclasses, bottomfacelabel, instancefaces

PartsNamesResult, FaceIdsResult, FaceClasses, BottomFaceLabels, InstanceFacesIds = ReadTxt(resultfile)
PointDatas ,PartNames, FaceIds = load_h5(h5path)

Sementicxs = []; Sementicys = []; Sementiczs = []
BottomFacecxs = []; BottomFacecys =[]; BottomFaceczs =[]
Instacncexs = []; Instacnceys = []; Instacncezs = []

Sementicx_1 = []; Sementicy_1 = []; Sementicz_1 = []
Sementicx_2 = []; Sementicy_2 = []; Sementicz_2 = []
Sementicx_3 = []; Sementicy_3 = []; Sementicz_3 = []
Sementicx_4 = []; Sementicy_4 = []; Sementicz_4 = []

BottomFacecx_1 = []; BottomFacecy_1 =[]; BottomFacecz_1 =[]
BottomFacecx_2 = []; BottomFacecy_2 =[]; BottomFacecz_2 =[]

partnum = partname[4:]
partnumfloat = float(partnum)
if partnumfloat in PartNames:
    if partnumfloat in PartNames:
        PartNamesList = PartNames.tolist()
        indexofpartnum = PartNamesList.index(partnumfloat)
        for i in range(len(FaceClasses[indexofpartnum])):
            if FaceClasses[indexofpartnum][i] == "10000": # Slot
                TempFaceId = float(FaceIdsResult[indexofpartnum][i])
                FaceIdsList = FaceIds.tolist()
                n = FaceIdsList[indexofpartnum].index(TempFaceId)
                PointDatasList = PointDatas.tolist()
                for j in range(len(PointDatasList[indexofpartnum][n])):
                    Sementicx_1.append(PointDatasList[indexofpartnum][n][j][0])
                    Sementicy_1.append(PointDatasList[indexofpartnum][n][j][1])
                    Sementicz_1.append(PointDatasList[indexofpartnum][n][j][2])
            elif FaceClasses[indexofpartnum][i] == "01000": # Base
                TempFaceId = float(FaceIdsResult[indexofpartnum][i])
                FaceIdsList = FaceIds.tolist()
                n = FaceIdsList[indexofpartnum].index(TempFaceId)
                PointDatasList = PointDatas.tolist()
                for j in range(len(PointDatasList[indexofpartnum][n])):
                    Sementicx_2.append(PointDatasList[indexofpartnum][n][j][0])
                    Sementicy_2.append(PointDatasList[indexofpartnum][n][j][1])
                    Sementicz_2.append(PointDatasList[indexofpartnum][n][j][2])
            elif FaceClasses[indexofpartnum][i] == "00100": # Pocket
                TempFaceId = float(FaceIdsResult[indexofpartnum][i])
                FaceIdsList = FaceIds.tolist()
                n = FaceIdsList[indexofpartnum].index(TempFaceId)
                PointDatasList = PointDatas.tolist()
                for j in range(len(PointDatasList[indexofpartnum][n])):
                    Sementicx_3.append(PointDatasList[indexofpartnum][n][j][0])
                    Sementicy_3.append(PointDatasList[indexofpartnum][n][j][1])
                    Sementicz_3.append(PointDatasList[indexofpartnum][n][j][2])
            elif FaceClasses[indexofpartnum][i] == "00010": # Hole
                TempFaceId = float(FaceIdsResult[indexofpartnum][i])
                FaceIdsList = FaceIds.tolist()
                n = FaceIdsList[indexofpartnum].index(TempFaceId)
                PointDatasList = PointDatas.tolist()
                for j in range(len(PointDatasList[indexofpartnum][n])):
                    Sementicx_4.append(PointDatasList[indexofpartnum][n][j][0])
                    Sementicy_4.append(PointDatasList[indexofpartnum][n][j][1])
                    Sementicz_4.append(PointDatasList[indexofpartnum][n][j][2])

        Sementicxs.append(Sementicx_1)
        Sementicxs.append(Sementicx_2)
        Sementicxs.append(Sementicx_3)
        Sementicxs.append(Sementicx_4)

        Sementicys.append(Sementicy_1)
        Sementicys.append(Sementicy_2)
        Sementicys.append(Sementicy_3)
        Sementicys.append(Sementicy_4)

        Sementiczs.append(Sementicz_1)
        Sementiczs.append(Sementicz_2)
        Sementiczs.append(Sementicz_3)
        Sementiczs.append(Sementicz_4)

        for i in range(len(BottomFaceLabels[indexofpartnum])):
            if BottomFaceLabels[indexofpartnum][i] == "0":
                TempFaceId = float(FaceIdsResult[indexofpartnum][i])
                FaceIdsList = FaceIds.tolist()
                n = FaceIdsList[indexofpartnum].index(TempFaceId)
                PointDatasList = PointDatas.tolist()
                for j in range(len(PointDatasList[indexofpartnum][n])):
                    BottomFacecx_1.append(PointDatasList[indexofpartnum][n][j][0])
                    BottomFacecy_1.append(PointDatasList[indexofpartnum][n][j][1])
                    BottomFacecz_1.append(PointDatasList[indexofpartnum][n][j][2])
            else:
                TempFaceId = float(FaceIdsResult[indexofpartnum][i])
                FaceIdsList = FaceIds.tolist()
                n = FaceIdsList[indexofpartnum].index(TempFaceId)
                PointDatasList = PointDatas.tolist()
                for j in range(len(PointDatasList[indexofpartnum][n])):
                    if (PointDatasList[indexofpartnum][n][j][0] != 0.0) & (PointDatasList[indexofpartnum][n][j][1] != 0.0) & \
                            (PointDatasList[indexofpartnum][n][j][2] != 0.0):
                        BottomFacecx_2.append(PointDatasList[indexofpartnum][n][j][0])
                        BottomFacecy_2.append(PointDatasList[indexofpartnum][n][j][1])
                        BottomFacecz_2.append(PointDatasList[indexofpartnum][n][j][2])
        BottomFacecxs.append(BottomFacecx_1)
        BottomFacecxs.append(BottomFacecx_2)
        BottomFacecys.append(BottomFacecy_1)
        BottomFacecys.append(BottomFacecy_2)
        BottomFaceczs.append(BottomFacecz_1)
        BottomFaceczs.append(BottomFacecz_2)

        for i in range(len(InstanceFacesIds[indexofpartnum])):
            EachInstanceFacesIds = []
            EachInstanceFacesId = []
            for instancesfacesid in InstanceFacesIds[indexofpartnum]:
                if instancesfacesid!="**":
                    EachInstanceFacesId.append(instancesfacesid)
                else:
                    EachInstanceFacesIds.append(EachInstanceFacesId)
                    EachInstanceFacesId = []

        for eachinstancefaceid in EachInstanceFacesIds:
            Instacncex = []
            Instacncey = []
            Instacncez = []
            for faceid in eachinstancefaceid:
                FaceIdsList = FaceIds.tolist()
                n = FaceIdsList[indexofpartnum].index(float(faceid))
                for j in range(len(PointDatasList[indexofpartnum][n])):
                    Instacncex.append(PointDatasList[indexofpartnum][n][j][0])
                    Instacncey.append(PointDatasList[indexofpartnum][n][j][1])
                    Instacncez.append(PointDatasList[indexofpartnum][n][j][2])
            Instacncexs.append(Instacncex)
            Instacnceys.append(Instacncey)
            Instacncezs.append(Instacncez)

        Face_Colors = ['gray', 'peru', 'wheat', 'green', 'red', 'blue', 'purple', 'pink', 'olive', 'yellow', 'brown',
                       'lawngreen', 'coral', 'sage', 'cyan']
        ClassLabels = ['Slot', 'Base', 'Pocket', 'Hole']
        BottomFaceLabels = ['Non-bottom face', 'Bottom face']
        Instancelabels = []
        for i in range(len(Instacncexs)):
            Instancelabels.append("Instance"+str(i))

        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(len(Sementicxs)):
            ax.plot(Sementicxs[i], Sementicys[i], Sementiczs[i], 'bo', color=Face_Colors[i], markersize=2, label=ClassLabels[i])
        plt.title('sementic_segmentation')
        plt.legend(ClassLabels)
        plt.show()
        plt.close()

        fig1 = plt.figure()
        ax1 = Axes3D(fig1)
        for i in range(len(BottomFacecxs)):
            ax1.plot(BottomFacecxs[i], BottomFacecys[i], BottomFaceczs[i], 'bo', color=Face_Colors[i], markersize=2, label=BottomFaceLabels[i])
        plt.title('bottom_face_identification')
        plt.legend(BottomFaceLabels)
        plt.show()
        plt.close()

        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        for i in range(len(Instacncexs)):
            ax2.plot(Instacncexs[i], Instacnceys[i], Instacncezs[i], 'bo', color=Face_Colors[i], markersize=2, label=Instancelabels[i])
        plt.title('instacne_segmentation')
        plt.legend(Instancelabels)
        plt.show()
        plt.close()
    else:
        print("The part is not in the .h5 file!")
else:
    print("The part is not in the results file!")






