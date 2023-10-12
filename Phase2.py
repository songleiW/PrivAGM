import random
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import sys
import threading
import math
import itertools
import fileIO
import mpcUtil
NUMUSER=76244   #twitter:76244
NUMFEAT=38      #twitter:38
MAXDEGREE=math.floor(NUMUSER/20)
epsilon_degree=1
epsilon_feat=1
epsilon_edgefeat=1
epsilon_numedge=1
epsilon_feat=1
epsilon_triangle=1
sensitivity_degree=2
sensitivity_feat=2
sensitivity_edgefeat=2
sensitivity_triangle=MAXDEGREE
sensitivity_numedge=1


def secShuffle(T1,T2):


    # P3 distributes correlation quantities to P1 and P2:
    A1=np.random.randint(np.iinfo(np.int32).max, size=(len(T1),2),dtype=np.int32)   #P1
    A2=np.random.randint(np.iinfo(np.int32).max, size=(len(T1),2),dtype=np.int32)   #P2
    B=np.random.randint(np.iinfo(np.int32).max, size=(len(T1),2),dtype=np.int32)    #P1
    pi1=np.random.permutation(len(T1))  #P1
    pi2=np.random.permutation(len(T1))  #P2
    delta=(A2[pi1]+A1)[pi2]+B #P1

    #P2 sends Z2 to P1:
    Z2=T2-A2

    #P1 sends Z1 to P1:
    Z1=(Z2+T1)[pi1]-A1



    return B, Z1[pi2]+delta

def secGetNoisyhistDegree(EdgesP1,EdgesP2,EdgesP3):
    inDegrees1 = np.zeros(NUMUSER, dtype=np.int32)
    inDegrees2 = np.zeros(NUMUSER, dtype=np.int32)
    inDegrees3 = np.zeros(NUMUSER, dtype=np.int32)

    outDegrees1 = np.zeros(NUMUSER, dtype=np.int32)
    outDegrees2 = np.zeros(NUMUSER, dtype=np.int32)
    outDegrees3 = np.zeros(NUMUSER, dtype=np.int32)

    #P1:
    for i in range(NUMUSER) :
        edge = EdgesP1[i]
        for j in range(len(edge)) :
            inDegrees1[edge[j][0]] += edge[j][1]
            outDegrees1[i] += edge[j][1]

    # P2:
    for i in range(NUMUSER) :
        edge = EdgesP2[i]
        for j in range(len(edge)) :
            inDegrees2[edge[j][0]] += edge[j][1]
            outDegrees2[i] += edge[j][1]

    # P3:
    for i in range(NUMUSER) :
        edge = EdgesP3[i]
        for j in range(len(edge)) :
            inDegrees3[edge[j][0]] += edge[j][1]
            outDegrees4[i] += edge[j][1]

    # P3 sends its secret share to P1 and P2:

    R1=np.random.randint(np.iinfo(np.int32).max, size=NUMUSER,dtype=np.int32)
    R2=np.random.randint(np.iinfo(np.int32).max, size=NUMUSER,dtype=np.int32)
    inDegrees1+=inDegrees3-R1
    inDegrees2+=R1
    outDegrees1 += inDegrees3 - R2
    outDegrees2 += R3

    dummyInDegrees1=[]
    dummyInDegrees2=[]
    dummyOutDegrees1=[]
    dummyOutDegrees2=[]
    Flag1=[]
    Flag2=[]

    for i in range(NUMUSER):
        Flag1.append(1)
        Flag2.append(0)



    # P3 sends dummy degrees to P1 and P2:
    mu = -1 * Delta * (math.log((np.exp(epsilon_degree / sensitivity_degree) + 1) * (1 - np.sqrt(1 - sensitivity_degree)))) / epsilon_degree
    dummyNum=np.rint(np.random.laplace(mu, sensitivity_degree / epsilon_degree, size=(MAXDEGREE, MAXDEGREE))).astype(np.int32)
    for i in range(MAXDEGREE):
        for j in range(MAXDEGREE):
            num=dummyNum[i,j]
            for n in range(num):
                r=np.random.randint(np.iinfo(np.int32).max,dtype=np.int32)
                Flag1.append(r)
                Flag2.append(0-r)
                r1=np.random.randint(np.iinfo(np.int32).max,dtype=np.int32)
                r2=np.random.randint(np.iinfo(np.int32).max,dtype=np.int32)

                dummyOutDegrees1.append(i-r1)
                dummyOutDegrees2.append(r1)
                dummyInDegrees1.append(j=r2)
                dummyInDegrees2.append(r2)

    inDegrees1=np.concatenate((inDegrees1,np.array(dummyInDegrees1)),axis=1)
    inDegrees2=np.concatenate((inDegrees2,np.array(dummyInDegrees2)),axis=1)
    outDegrees1=np.concatenate((outDegrees1,np.array(dummyOutDegrees1)),axis=1)
    outDegrees2=np.concatenate((outDegrees2,np.array(dummyOutDegrees2)),axis=1)


    T1,T2=secShuffle(np.concatenate((inDegrees1, outDegrees1,np.array(Flag1)), axis=0),
                                 np.concatenate((inDegrees2, outDegrees2,np.array(Flag2)), axis=0))

    inDegrees1=T1[:,0]
    outDegrees1=T1[:,1]
    Flag1=T1[:,2]

    inDegrees2 = T2[:, 0]
    outDegrees2 = T2[:, 1]
    Flag2 = T2[:, 2]





    # P1 sends secret shares of (i,j) to P2 to reveal the indexes:
    inDegrees=inDegrees1+inDegrees2
    outDegrees=outDegrees1+outDegrees2



    histDegree1=np.zeros((MAXDEGREE,MAXDEGREE),dtype=np.int32)
    histDegree2=np.zeros((MAXDEGREE,MAXDEGREE),dtype=np.int32)



    # Counting
    for i in range(len(inDegrees)) :
        if outDegrees[i]<MAXDEGREE and inDegrees[i]<MAXDEGREE:
            histDegree1[outDegrees[i],inDegrees[i]]+=Flag1[i] #"%MAXDEGREE" to clip degree
            histDegree2[outDegrees[i],inDegrees[i]]+=Flag2[i] #"%MAXDEGREE" to clip degree



    #Add discrete Lap noises
    noise=np.rint(np.random.laplace(0, sensitivity_degree / epsilon_degree,size=(MAXDEGREE,MAXDEGREE))).astype(np.int32)
    R=np.random.randint(np.iinfo(np.int32).max, size=(MAXDEGREE,MAXDEGREE),dtype=np.int32)
    NoisyhistDegree1=histDegree1+R
    NoisyhistDegree2=histDegree2-R+noise

    return NoisyhistDegree+NoisyhistDegree2

def getFeature(FeaturesP1,FeaturesP2,FeaturesP3):

    histFeature1=np.zeros(NUMFEAT,dtype=np.int32)
    histFeature2=np.zeros(NUMFEAT,dtype=np.int32)
    histFeature3=np.zeros(NUMFEAT,dtype=np.int32)




    # P3 sends its secret share to P1 and P2:

    R1 = np.random.randint(np.iinfo(np.int32).max, size=NUMUSER, dtype=np.int32)
    FeaturesP1 += FeaturesP3 - R1
    FeaturesP2 += R1



    Flag1 = []
    Flag2 = []
    for i in range(NUMUSER) :
        Flag1.append(1)
        Flag2.append(0)



    # P3 sends dummy features to P1 and P2:
    dumFeat1=[]
    dumFeat2=[]
    mu = -1 * Delta * (math.log(
        (np.exp(epsilon_degree / sensitivity_feat) + 1) * (1 - np.sqrt(1 - sensitivity_feat)))) / epsilon_feat
    dummyNum = np.rint(np.random.laplace(mu, sensitivity_feat / epsilon_feat, size=NUMFEAT)).astype(np.int32)
    for i in range(NUMFEAT):
        num = dummyNum[i]
        for n in range(num) :
            r = np.random.randint(np.iinfo(np.int32).max, dtype=np.int32)
            Flag1.append(r)
            Flag2.append(0 - r)
            r1 = np.random.randint(np.iinfo(np.int32).max, dtype=np.int32)

            dumFeat1.append(i - r1)
            dumFeat2.append(r1)


    FeaturesP1 = np.concatenate((FeaturesP1, np.array(dumFeat1)), axis=1)
    FeaturesP2 = np.concatenate((FeaturesP2, np.array(dumFeat2)), axis=1)

    T1,T2=secShuffle(np.concatenate((FeaturesP1,np.array(Flag1)), axis=0),
                     np.concatenate((FeaturesP2,np.array(Flag2)), axis=0))




    FeaturesP1 = T1[:, 0]
    Flag1 = T1[:, 1]

    FeaturesP2 = T2[:, 0]
    Flag2 = T2[:, 1]



    # P1 sends secret shares of feature to P2 to reveal the indexes:
    Features = FeaturesP1 + FeaturesP2


    histFeature=np.zeros(NUMFEAT,dtype=np.int32)

    # Counting
    for i in range(len(FeaturesP1)) :
        histFeature1[Features[i]]+=Flag1[i]
        histFeature2[Features[i]]+=Flag2[i]



    # Add discrete Lap noises
    noise = np.rint(np.random.laplace(0, sensitivity_feat / epsilon_feat, size=NUMFEAT)).astype(
        np.int32)
    R = np.random.randint(np.iinfo(np.int32).max, size=NUMFEAT, dtype=np.int32)

    histFeature1 = histFeature1 + R
    histFeature2 = histFeature2 - R + noise

    #Reveal histogram of features:
    NoisyhistFeature=histFeature1+histFeature2
    NoisyhistFeature[NoisyhistFeature<0]=0


    return NoisyhistFeature

def secGetNoisyFeatureEdge(EdgesP1,EdgesP2,EdgesP3,FeaturesP1,FeaturesP2,FeaturesP3):


    #Contructing the encrypted feature pairs:
    FeaturePairsP1=[]
    FeaturePairsP2=[]
    FlagsP1 = []
    FlagsP2 = []

    #P1-P2:
    for i in range(NUMUSER):
        for j in len(EdgesP1[i]):
            if EdgesP1[i][j][2]==2:
                FeaturePairsP1.append([FeaturesP1[0][i], FeaturesP1[0][EdgesP1[i][j][0]]])
                FlagsP1.append(EdgesP1[i][j][1])

    #P2-P1:
    for i in range(NUMUSER):
        for j in len(EdgesP2[i]):
            if EdgesP2[i][j][2]==1:
                FeaturePairsP2.append([FeaturesP2[0][i], FeaturesP2[0][EdgesP2[i][j][0]]])
                FlagsP2.append(EdgesP2[i][j][1])


    # P3-P1:
    for i in range(NUMUSER):
        for j in len(EdgesP3[i]):
            if EdgesP3[i][j][2]==1:
                FeaturePairsP2.append([FeaturesP3[0][i], FeaturesP3[0][EdgesP3[i][j][0]]])
                FlagsP2.append(EdgesP3[i][j][1])

    #P1-P3:
    for i in range(NUMUSER):
        for j in len(EdgesP1[i]):
            if EdgesP1[i][j][2]==3:
                FeaturePairsP1.append([FeaturesP1[0][i], FeaturesP1[0][EdgesP1[i][j][0]]])
                FlagsP1.append(EdgesP1[i][j][1])

    # P3-P2:
    for i in range(NUMUSER) :
        for j in len(EdgesP3[i]) :
            if EdgesP3[i][j][2] == 2 :
                FeaturePairsP1.append([FeaturesP3[0][i], FeaturesP3[0][EdgesP3[i][j][0]]])
                FlagsP1.append(EdgesP3[i][j][1])

    # P2-P3:
    for i in range(NUMUSER) :
        for j in len(EdgesP2[i]) :
            if EdgesP3[i][j][2] == 3 :
                FeaturePairsP2.append([FeaturesP2[0][i], FeaturesP2[0][EdgesP3[i][j][0]]])
                FlagsP2.append(EdgesP2[i][j][1])

    # P3 sends dummy featuresPair to P1 and P2:
    dumFeat1 = []
    dumFeat2 = []
    mu = -1 * Delta * (math.log(
        (np.exp(epsilon_edgefeat / sensitivity_edgefeat) + 1) * (1 - np.sqrt(1 - sensitivity_edgefeat)))) / epsilon_edgefeat
    dummyNum = np.rint(np.random.laplace(mu, sensitivity_edgefeat / epsilon_feat, size=NUMFEAT)).astype(np.int32)
    for i in range(NUMFEAT*NUMFEAT) :
        num = dummyNum[i]
        for n in range(num) :
            r = np.random.randint(np.iinfo(np.int32).max, dtype=np.int32)
            FlagsP1.append(r)
            FlagsP2.append(0 - r)
            r1 = np.random.randint(np.iinfo(np.int32).max, dtype=np.int32)

            dumFeat1.append(i - r1)
            dumFeat2.append(r1)

    FeaturePairsP1 = np.concatenate((FeaturePairsP1, np.array(dumFeat1)), axis=1)
    FeaturePairsP2 = np.concatenate((FeaturePairsP2, np.array(dumFeat2)), axis=1)

    T1, T2 = secShuffle(np.concatenate((FeaturePairsP1, np.array(FlagsP1)), axis=0),
                        np.concatenate((FeaturePairsP2, np.array(FlagsP2)), axis=0))

    FeaturesP1 = T1[:, 0]
    Flag1 = T1[:, 1]

    FeaturesP2 = T2[:, 0]
    Flag2 = T2[:, 1]

    # P1 sends secret shares of feature to P2 to reveal the indexes:
    Features = FeaturesP1 + FeaturesP2


    histFeatureEdge1 = np.zeros((NUMFEAT, NUMFEAT), dtype=np.int32)
    histFeatureEdge2 = np.zeros((NUMFEAT, NUMFEAT), dtype=np.int32)


    # Counting
    for i in range(NUMUSER) :
        feat1 = Features[i]
        for j in range(len(Edges[i])) :
            feat2 = Features[Edges[i][j]]
            histFeatureEdge1[feat1, feat2] += Flag1[i,j]
            histFeatureEdge2[feat1, feat2] += Flag2[i,j]


    # Add discrete Lap noises
    noise = np.rint(np.random.laplace(0, sensitivity_edgefeat / epsilon_edgefeat, size=NUMFEAT)).astype(
        np.int32)
    R = np.random.randint(np.iinfo(np.int32).max, size=NUMFEAT, dtype=np.int32)

    histFeatureEdge1 += R
    histFeatureEdge2 = histFeatureEdge2 - R + noise

    # Reveal histogram of features:
    NoisyhistFeatureEdge = histFeatureEdge1 + histFeatureEdge2
    NoisyhistFeatureEdge[NoisyhistFeatureEdge < 0] = 0

    return NoisyhistFeatureEdge

def secGetNoisyTriangle(EdgesP1,EdgesP2,EdgesP3):

    # Contructing the encrypted edges:
    Edges1 = []
    Edges2 = []
    Edges3 = []

    FlagsP1 = []
    FlagsP2 = []
    FlagsP3 = []



    NumTriangle11 = 0
    NumTriangle12 = 0


    # P1-P2:
    for i in range(NUMUSER) :
        for j in len(EdgesP1[i]) :
            if EdgesP1[i][j][2] == 2 :
                Edges1.append([i,EdgesP1[i][j][0]])
                FlagsP1.append(EdgesP1[i][j][1])

    # P2-P1:
    for i in range(NUMUSER) :
        for j in len(EdgesP2[i]) :
            if EdgesP2[i][j][2] == 1 :
                Edges2.append([i, EdgesP2[i][j][0]])
                FlagsP2.append(EdgesP2[i][j][1])


    for i in range(NUMUSER) :
        for j in Edges1[i]:
            if j <i:
                for k in Edges1[j]:
                    if k<i and i in Edges1[k] :
                        a,b= mpcUtil.mpcMulti(FlagsP1[i],FlagsP1[j],FlagsP2[i],FlagsP2[j])
                        NumTriangle11+=a
                        NumTriangle12+=b




    Edges2 = []
    Edges3 = []

    FlagsP2 = []
    FlagsP3 = []

    # P2-P3:

    for i in range(NUMUSER) :
        for j in len(EdgesP2[i]) :
            if EdgesP2[i][j][2] == 3 :
                Edges2.append([i, EdgesP2[i][j][0]])
                FlagsP2.append(EdgesP2[i][j][1])

    # P3-P2:
    for i in range(NUMUSER) :
        for j in len(EdgesP3[i]) :
            if EdgesP3[i][j][2] == 2 :
                Edges3.append([i, EdgesP3[i][j][0]])
                FlagsP3.append(EdgesP3[i][j][1])

    for i in range(NUMUSER) :
        for j in Edges2[i] :
            if j < i :
                for k in Edges2[j] :
                    if k < i and i in Edges2[k] :
                        a,b= mpcUtil.mpcMulti(FlagsP2[i], FlagsP2[j], FlagsP3[i], FlagsP3[j])
                        NumTriangle11 += a
                        NumTriangle12 += b

    Edges3 = []
    Edges1 = []

    FlagsP3 = []
    FlagsP1 = []

    # P3-P1:

    for i in range(NUMUSER) :
        for j in len(EdgesP3[i]) :
            if EdgesP3[i][j][2] == 1 :
                Edges3.append([i, EdgesP3[i][j][0]])
                FlagsP3.append(EdgesP3[i][j][1])

    # P1-P3:
    for i in range(NUMUSER) :
        for j in len(EdgesP1[i]) :
            if EdgesP1[i][j][2] == 3 :
                Edges1.append([i, EdgesP1[i][j][0]])
                FlagsP1.append(EdgesP1[i][j][1])

    for i in range(NUMUSER) :
        for j in Edges3[i] :
            if j < i :
                for k in Edges3[j] :
                    if k < i and i in Edges3[k] :
                        a, b = mpcUtil.mpcMulti(FlagsP3[i], FlagsP3[j], FlagsP1[i], FlagsP1[j])
                        NumTriangle11 += a
                        NumTriangle12 += b



    # Add discrete Lap noises
    noise = np.rint(np.random.laplace(0, sensitivity_edgefeat / epsilon_edgefeat, size=NUMFEAT)).astype(
        np.int32)
    R = np.random.randint(np.iinfo(np.int32).max, size=NUMFEAT, dtype=np.int32)

    NumTriangle11 += R
    NumTriangle12 = NumTriangle12 - R + noise

    # Reveal histogram of features:
    NumTriangle1 = NumTriangle11 + NumTriangle12
    NumTriangle1*=9







    # Contructing the encrypted edges:
    Edges1 = []
    Edges2 = []
    Edges3 = []

    FlagsP1 = []
    FlagsP2 = []
    FlagsP3 = []


    # P1-P2:
    for i in range(NUMUSER) :
        for j in len(EdgesP1[i]) :
            if EdgesP1[i][j][2] == 2 :
                Edges1.append([i, EdgesP1[i][j][0]])
                FlagsP1.append(EdgesP1[i][j][1])

    # P2-P1:
    for i in range(NUMUSER) :
        for j in len(EdgesP2[i]) :
            if EdgesP2[i][j][2] == 1 :
                Edges2.append([i, EdgesP2[i][j][0]])
                FlagsP2.append(EdgesP2[i][j][1])


    for i in range(NUMUSER):
        combinations = list(itertools.combinations(Edges1[i], 2))
        for combo in combinations:
            if combo[0] in Edges1[combo[1]]:
                a, b = mpcUtil.mpcMulti(FlagsP1[i], FlagsP1[combo[0]], FlagsP2[i], FlagsP2[combo[0]])
                NumTriangle21 += a
                NumTriangle22 += b
            if combo[1] in Edges1[combo[0]]:
                a, b = mpcUtil.mpcMulti(FlagsP1[i], FlagsP1[combo[1]], FlagsP2[i], FlagsP2[combo[1]])
                NumTriangle21 += a
                NumTriangle22 += b


    Edges2 = []
    Edges3 = []

    FlagsP2 = []
    FlagsP3 = []

    # P2-P3:

    for i in range(NUMUSER) :
        for j in len(EdgesP2[i]) :
            if EdgesP2[i][j][2] == 3 :
                Edges2.append([i, EdgesP2[i][j][0]])
                FlagsP2.append(EdgesP2[i][j][1])

    # P3-P2:
    for i in range(NUMUSER) :
        for j in len(EdgesP3[i]) :
            if EdgesP3[i][j][2] == 2 :
                Edges3.append([i, EdgesP3[i][j][0]])
                FlagsP3.append(EdgesP3[i][j][1])

    for i in range(NUMUSER) :
        combinations = list(itertools.combinations(Edges1[i], 2))
        for combo in combinations :
            if combo[0] in Edges2[combo[1]] :
                a, b = mpcUtil.mpcMulti(FlagsP2[i], FlagsP2[combo[0]], FlagsP3[i], FlagsP3[combo[0]])
                NumTriangle21 += a
                NumTriangle22 += b
            if combo[1] in Edges2[combo[0]] :
                a, b = mpcUtil.mpcMulti(FlagsP2[i], FlagsP2[combo[1]], FlagsP3[i], FlagsP3[combo[1]])
                NumTriangle21 += a
                NumTriangle22 += b

    Edges3 = []
    Edges1 = []

    FlagsP3 = []
    FlagsP1 = []

    # P3-P1:

    for i in range(NUMUSER) :
        for j in len(EdgesP3[i]) :
            if EdgesP3[i][j][2] == 1 :
                Edges3.append([i, EdgesP3[i][j][0]])
                FlagsP3.append(EdgesP3[i][j][1])

    # P1-P3:
    for i in range(NUMUSER) :
        for j in len(EdgesP1[i]) :
            if EdgesP1[i][j][2] == 3 :
                Edges1.append([i, EdgesP1[i][j][0]])
                FlagsP1.append(EdgesP1[i][j][1])

    for i in range(NUMUSER) :
        combinations = list(itertools.combinations(Edges1[i], 2))
        for combo in combinations :
            if combo[0] in Edges3[combo[1]] :
                a, b = mpcUtil.mpcMulti(FlagsP3[i], FlagsP3[combo[0]], FlagsP1[i], FlagsP1[combo[0]])
                NumTriangle21 += a
                NumTriangle22 += b
            if combo[1] in Edges3[combo[0]] :
                a, b = mpcUtil.mpcMulti(FlagsP3[i], FlagsP3[combo[1]], FlagsP1[i], FlagsP1[combo[1]])
                NumTriangle21 += a
                NumTriangle22 += b


    # Add discrete Lap noises
    noise = np.rint(np.random.laplace(0, sensitivity_edgefeat / epsilon_edgefeat, size=NUMFEAT)).astype(
        np.int32)
    R = np.random.randint(np.iinfo(np.int32).max, size=NUMFEAT, dtype=np.int32)

    NumTriangle21 += R
    NumTriangle22 = NumTriangle22 - R + noise

    # Reveal histogram of features:
    NumTriangle2 = NumTriangle21 + NumTriangle22
    NumTriangle2 *= 9



    return [NumTriangle1,NumTriangle2]

def secGetNumEdges(EdgesP1,EdgesP2,EdgesP3):
    num1=0
    num2=0
    num3=0
    for i in range(NUMUSER):
        for j in len(EdgesP1[i]):
            num1+=EdgesP1[i][j][2]

    for i in range(NUMUSER) :
        for j in len(EdgesP2[i]) :
            num2 += EdgesP2[i][j][2]

    for i in range(NUMUSER):
        for j in len(EdgesP3[i]):
            num3+=EdgesP3[i][j][2]



    # P3 sends its secret share to P1 and P2:

    R1 = np.random.randint(np.iinfo(np.int32).max, dtype=np.int32)
    num1 += num3 - R1
    num2 += R1

    # Add discrete Lap noises
    noise = np.rint(np.random.laplace(0, sensitivity_numedge / epsilon_numedge)).astype(np.int32)
    R=np.random.randint(np.iinfo(np.int32).max,dtype=np.int32)

    num1 += noise-R
    num2+=noise+R

    return num1+num2

def secExtractGraph(flag):


    if flag == 1 :  # twitter dataset;
        directory = "CiphertextTwitter/"  # file path

    FeaturesP1=np.load(directory + "P1_feature.npy")
    FeaturesP2=np.load(directory + "P2_feature.npy")
    FeaturesP3=np.load(directory + "P3_feature.npy")
    EdgesP1=np.load(directory + "P1_edges.npy")
    EdgesP2=np.load(directory + "P2_edges.npy")
    EdgesP3=np.load(directory + "P3_edges.npy")



    NoisyhistDegree1=secGetNoisyhistDegree(EdgesP1,EdgesP2,EdgesP3)     #Securely extracting the histgram of degrees

    NoisyhistFeature=secGetFeature(FeaturesP1,FeaturesP2,FeaturesP3)    #Securely extracting the histgram of features

    NoisyhistFeatureEdge=secGetNoisyFeatureEdge(EdgesP1,EdgesP2,EdgesP3,FeaturesP1,FeaturesP2,FeaturesP3)   #Securely extracting the histgram of feature pairs

    NoisyTriangle=secGetNoisyTriangle(EdgesP1,EdgesP2,EdgesP3)  #Securely extracting the number of tringales

    NoisynumEdges=secGetNumEdges(EdgesP1,EdgesP2,EdgesP3)   #Securely extracting the number of edges



    #Writing the extracted features to the .npy file
    if flag == 1 :   #twitter dataset
        directory = "GraphFeatures/twitter/" # file path

    np.save(directory+"histDegree1.npy",NoisyhistDegree)
    np.save(directory+"OriginalhistDegree1.npy",histDegree)
    np.save(directory+"histFeature1.npy",NoisyhistFeature)
    np.save(directory+"histFeatureEdge1.npy",NoisyhistFeatureEdge)
    np.save(directory+"Triangle1.npy",NoisyTriangle)
    np.save(directory+"NoisynumEdges1.npy",NoisynumEdges)



