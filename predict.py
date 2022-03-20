import os
import h5py
import numpy as np
from sklearn.cluster import estimate_bandwidth
from tensorflow.python.keras import backend as K
from models import cluster, model
import time
import tensorflow as tf



def load_h5_name_id(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][1:2]
    nameofpart = f['name'][1:2]
    id = f['id'][1:2]
    return (data ,nameofpart, id)

with tf.device('/CPU'):
    path=r'demo_datas'
    model_save_weights_file=r'models\ASIN_best_weights.h5'
    WriteFile=r'draw\Results.txt'

    filenames_predict=[]
    for d in os.listdir(path):
        filenames_predict.append(d)

    predict_points_OnFace = None
    predict_namesofpart = None
    predict_FacesId = None

    for d in filenames_predict:
        cur_points_OnFace, NamesofPart, FacesID = load_h5_name_id(os.path.join(path, d))
        if predict_points_OnFace is None:
            predict_points_OnFace = cur_points_OnFace
            predict_namesofpart = NamesofPart
            predict_FacesId = FacesID
        else:
            predict_points_OnFace = np.vstack((predict_points_OnFace, cur_points_OnFace))
            predict_namesofpart = np.hstack((predict_namesofpart, NamesofPart))
            predict_FacesId = np.hstack((predict_FacesId, FacesID), dtype=int)
    predict_namesofpart = predict_namesofpart.reshape((len(predict_namesofpart), 1))

    time_start=time.time()
    loaded_model = model.ASIN_model()
    #loaded_model.summary()
    loaded_model.load_weights(model_save_weights_file)

    results = loaded_model.predict(predict_points_OnFace)


    r_segments = results[0]
    r_bottomface = results[1]
    r_instances = results[2]


    deduction_layers = loaded_model.get_layer('OUTPUT2-1')
    embedding_layers = loaded_model.get_layer('OUTPUT2-64')
    input_layers = loaded_model.input
    embedding_layers_outputs = []
    deduction_layers_outputs = []


    for i in range(r_instances.shape[0]):
        iterate = K.function([input_layers], [embedding_layers.output[i]])
        iterate2 = K.function([input_layers], [deduction_layers.output[i]])
        embedding_layers_output = iterate([predict_points_OnFace])
        deduction_layers_output = iterate2([predict_points_OnFace])
        embedding_layers_outputs.append(embedding_layers_output)
        deduction_layers_outputs.append(deduction_layers_output)
        #time_end = time.time()


    #print(embedding_layers_outputs)

    logFile = open(WriteFile, 'w', encoding="utf-8-sig")
    logFile.write('start'+'\n')

    AllclustersResults = []
    ClusterResults = []
    ClusterSimilarityMatrix = []
    ModifyPredictResult = []

    for i in range(len(embedding_layers_outputs)):
        PartResultE = embedding_layers_outputs[i][0]
        PartResultD = deduction_layers_outputs[i][0]
        NewPartResultD = PartResultD
        for j in range(PartResultE.shape[1]):
            if j != 0:
                PartResultD = np.hstack((PartResultD, NewPartResultD))
        #PartResult = np.hstack((PartResultE, PartResultD))
        PartResult = (PartResultE+PartResultD)/2.0
        mycluster = cluster.ClusterMethod(PartResult)

        # Predict the bandwidth by adjusting quantile.
        # The value of quantile can be 0.14, 0.15, or 0.16, which may lead to a better result.

        bandwidth = estimate_bandwidth(PartResult, quantile=0.16)
        pred_gmm = mycluster.MeanShift(bandwidth)
        TrueFaceId = np.array(predict_FacesId[i], dtype=int)
        PredictSimilarMatrix = r_instances[i]

        PredictSimilarMatrixINT = np.eye(PredictSimilarMatrix.shape[0])
        for j in range(PredictSimilarMatrix.shape[0]):
            for k in range(PredictSimilarMatrix.shape[1]):
                if PredictSimilarMatrix[j][k] > 0.5:
                    PredictSimilarMatrixINT[j][k] = 1
                    PredictSimilarMatrixINT[k][j] = 1

        ClusterResult = []
        CRClasses = []
        for j in range(len(pred_gmm)):
            if j == 0:
                CRClasses.append(pred_gmm[j])
            else:
                if pred_gmm[j] not in CRClasses:
                    CRClasses.append(pred_gmm[j])
        for j in range(len(CRClasses)):
            CR = []
            CRClass = CRClasses[j]
            for k in range(len(pred_gmm)):
                if CRClass == pred_gmm[k]:
                    CR.append(TrueFaceId[k])
            ClusterResult.append(CR)
        AllclustersResults.append(ClusterResult)
        SimilarityMatrix = np.eye(len(pred_gmm))
        allindexes = []
        for Facesinoneinstance in ClusterResult:
            if len(Facesinoneinstance) > 1:
                indexes = []
                for Faceinoneinstance in Facesinoneinstance:
                    FaceinoneinstanceINT = int(Faceinoneinstance)
                    TrueFaceIdList = TrueFaceId.tolist()
                    indexes.append(TrueFaceIdList.index(Faceinoneinstance))
                for j in range(len(indexes)):
                    for k in range(j, len(indexes)):
                        SimilarityMatrix[indexes[j]][indexes[k]] = 1
                        SimilarityMatrix[indexes[k]][indexes[j]] = 1
                zeroid = []
                for j in range(len(TrueFaceId)):
                    if TrueFaceId[j] == 0:
                        zeroid.append(j)
                for j in range(len(zeroid)):
                    for k in range(j, len(zeroid)):
                        SimilarityMatrix[zeroid[j]][zeroid[k]] = 1
                        SimilarityMatrix[zeroid[k]][zeroid[j]] = 1
        NewPredictMatrix = np.zeros((SimilarityMatrix.shape[0], SimilarityMatrix.shape[1]))
        for j in range(SimilarityMatrix.shape[0]):
            for k in range(SimilarityMatrix.shape[1]):
                if (PredictSimilarMatrixINT[j][k] == 1) & (SimilarityMatrix[j][k] == 1):
                    NewPredictMatrix[j][k] = 1
        ClusterSimilarityMatrix.append(SimilarityMatrix)

        ModifyPredictResult.append(NewPredictMatrix)



    PredictResult_Instances = []
    for i in range(len(ModifyPredictResult)):
        TrueFaceId = predict_FacesId[i]
        SimilarMatrix = ModifyPredictResult[i]
        SimilarMatrixINT = np.round(SimilarMatrix)
        allindexsim = []
        for j in range(SimilarMatrix.shape[0]):
            indexsim = []
            flag = 0
            for k in range(SimilarMatrix.shape[1]):
                if round(SimilarMatrix[j][k]) == 1:
                    indexsim.append(k)
            if len(indexsim)>0:
                allindexsim.append(indexsim)
        instance_results=[]

        allindexofallindexsim = []
        MaxScores = []
        for j in range(SimilarMatrix.shape[1]):
            indexofallindexsim = []
            for myindexsim in allindexsim:
                if j in myindexsim:
                    indexofallindexsim.append(myindexsim)
            allindexofallindexsim.append(indexofallindexsim)

        allscoreofproposal = []
        MaxScoreFaceProposal = []
        yi = 0
        for myindexofallindexsim in allindexofallindexsim:
            scoreinonepropose = []
            for eveyindexes in myindexofallindexsim:
                sumscore = 0
                for eveyindex in eveyindexes:
                    if yi != eveyindex:
                        sumscore += SimilarMatrix[yi][eveyindex]
                avescore = sumscore / (len(eveyindexes))
                scoreinonepropose.append(avescore)
            yi += 1
            maxscore = max(scoreinonepropose)
            MaxScores.append(maxscore)
            indexofMaxScore = scoreinonepropose.index(maxscore)
            MaxScoreFaceProposal.append(myindexofallindexsim[indexofMaxScore])
            allscoreofproposal.append(scoreinonepropose)

        DiffentGroups = []
        for j in range(len(MaxScoreFaceProposal)):
            if j == 0:
                DiffentGroups.append(MaxScoreFaceProposal[j])
            else:
                if MaxScoreFaceProposal[j] not in DiffentGroups:
                    DiffentGroups.append(MaxScoreFaceProposal[j])

        IsolatedProposals = []
        IntersectProposals = []
        IntersectScores = []
        kk = 0
        for diffentgroup in DiffentGroups:
            num = 0
            for element in diffentgroup:
                for otherdiffentgroup in DiffentGroups:
                    if diffentgroup != otherdiffentgroup:
                        if element in otherdiffentgroup:
                            num += 1
                            break
            if num == 0:
                IsolatedProposals.append(diffentgroup)
            else:
                IntersectProposals.append(diffentgroup)
                IntersectScores.append(MaxScores[kk])
            kk += 1
        IntersectScoresCopy = IntersectScores.copy()
        IntersectScoresCopy.sort(reverse=True)
        indexesofIntersectScores = []
        for IntersectScore in IntersectScoresCopy:
            index = IntersectScores.index(IntersectScore)
            indexesofIntersectScores.append(index)

        kk = 0
        for index in indexesofIntersectScores:
            num = 0
            if kk == 0:
                IsolatedProposals.append(IntersectProposals[index])
            else:
                for proposal in IsolatedProposals:
                    for k in range(len(IntersectProposals[index])):
                        if IntersectProposals[index] != proposal:
                            if IntersectProposals[index][k] in proposal:
                                num = 1
                                break
                    if num == 1:
                        break
                if num == 0:
                    if IntersectProposals[index] not in IsolatedProposals:
                        IsolatedProposals.append(IntersectProposals[index])
            kk += 1

        for myindexsim in IsolatedProposals:
            instance_result = []
            for myindex in myindexsim:
                instance_result.append(TrueFaceId[myindex])
            instance_results.append(instance_result)
        PredictResult_Instances.append(instance_results)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    for i in range(predict_FacesId.shape[0]):
        logFile.write('Part' + str(int(predict_namesofpart[i])))
        logFile.write('\n')
        for j in range(predict_FacesId.shape[1]):
            logFile.write(str(int(predict_FacesId[i][j]))+'\t')
        logFile.write('\n')
        for j in range(r_segments.shape[1]):
            for k in range(r_segments.shape[2]):
                Label = round(r_segments[i][j][k])
                logFile.write(str(Label))
            logFile.write('\t')
        logFile.write('\n')
        for j in range(r_bottomface.shape[1]):
            for k in range(r_bottomface.shape[2]):
                BFLabel = round(r_bottomface[i][j][k])
                logFile.write(str(BFLabel))
            logFile.write('\t')
        logFile.write('\n')
        #上传版本不需要，但是有用
        # for j in range(len(AllclustersResults[i])):
        #     for k in range(len(AllclustersResults[i][j])):
        #         logFile.write(str(int(AllclustersResults[i][j][k]))+'\t')
        #     logFile.write('**'+'\t')
        # logFile.write('\n')
        for j in range(len(PredictResult_Instances[i])):
            for k in range(len(PredictResult_Instances[i][j])):
                logFile.write(str(int(PredictResult_Instances[i][j][k]))+'\t')
            logFile.write('**'+'\t')
        logFile.write('\n')
        logFile.write(str(len(PredictResult_Instances[i])-1))
        logFile.write('\n')
    logFile.write('end'+'\n')
    logFile.close()