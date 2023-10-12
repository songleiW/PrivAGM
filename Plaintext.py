import random
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import sys
import threading
import math
import itertools
import fileIO
from queue import Queue
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


def gethistDegree(Edges):



    histDegree = np.zeros((MAXDEGREE, MAXDEGREE), dtype=np.int32)

    inDegrees = np.zeros(NUMUSER, dtype=np.int32)
    outDegrees = np.zeros(NUMUSER, dtype=np.int32)

    for i in range(NUMUSER) :
        edge = Edges[i]
        for j in range(len(edge)) :
            inDegrees[edge[j]] += 1

    for i in range(NUMUSER) :
        outDegrees[i] = len(Edges[i])

    # Counting
    for i in range(NUMUSER) :
        if outDegrees[i] < MAXDEGREE and inDegrees[i] < MAXDEGREE :
            histDegree[outDegrees[i], inDegrees[i]] += 1  # "%MAXDEGREE" to clip degree


    histDegree = np.zeros((MAXDEGREE, MAXDEGREE), dtype=np.int32)

    inDegrees = np.zeros(NUMUSER, dtype=np.int32)
    outDegrees = np.zeros(NUMUSER, dtype=np.int32)

    for i in range(NUMUSER) :
        edge = Edges[i]
        for j in range(len(edge)) :
            inDegrees[edge[j]] += 1

    for i in range(NUMUSER) :
        outDegrees[i] = len(Edges[i])

    # Counting
    for i in range(NUMUSER) :
        if outDegrees[i] < MAXDEGREE and inDegrees[i] < MAXDEGREE :
            histDegree[outDegrees[i], inDegrees[i]] += 1  # "%MAXDEGREE" to clip degree

    return histDegree

def getNoisyhistDegree(Edges):


    histDegree=np.zeros((MAXDEGREE,MAXDEGREE),dtype=np.int32)
    NoisyhistDegree=np.zeros((MAXDEGREE,MAXDEGREE),dtype=np.int32)



    inDegrees=np.zeros(NUMUSER,dtype=np.int32)
    outDegrees=np.zeros(NUMUSER,dtype=np.int32)

    for i in range(NUMUSER):
        edge=Edges[i]
        for j in range(len(edge)):
            inDegrees[edge[j]]+=1

    for i in range(NUMUSER):
        outDegrees[i]=len(Edges[i])

    noisyInDegrees=inDegrees
    noisyOutDegrees=outDegrees

    for i in range(NUMUSER):
        noisyInDegrees[i]+=np.rint(np.random.laplace(0, sensitivity_degree / epsilon_degree))
        noisyOutDegrees[i]+=np.rint(np.random.laplace(0, sensitivity_degree / epsilon_degree))


    # Counting
    for i in range(NUMUSER) :
        if outDegrees[i] < MAXDEGREE and inDegrees[i] < MAXDEGREE :
            NoisyhistDegree[noisyInDegrees[i], noisyOutDegrees[i]] += 1  # "%MAXDEGREE" to clip degree



    # Counting
    for i in range(NUMUSER) :
        if outDegrees[i]<MAXDEGREE and inDegrees[i]<MAXDEGREE:
            histDegree[outDegrees[i],inDegrees[i]]+=1 #"%MAXDEGREE" to clip degree



    #print(hellinger_distance(NoisyhistDegree,histDegree))




    #Add discrete Lap noises
    NoisyhistDegree=histDegree+np.rint(np.random.laplace(0, sensitivity_degree / epsilon_degree,size=(MAXDEGREE,MAXDEGREE))).astype(np.int32)

    #NoisyhistDegree[NoisyhistDegree<3]=0 #offest to smooth the passivie value
    #NoisyhistDegree-=np.min(NoisyhistDegree)


    return NoisyhistDegree,histDegree

def hellinger_distance(p, q) :
    # Compute the Hellinger distance between two probability distributions
    # p and q are arrays or matrices representing the probability distributions

    # Ensure that p and q are normalized probability distributions
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)

    # Compute the Hellinger distance
    distance = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

    return distance

def getFeature(Features):


    histFeature=np.zeros(NUMFEAT,dtype=np.int32)
    # Counting
    for i in range(NUMUSER) :
        histFeature[Features[i]]=histFeature[Features[i]]+1

    # Add discrete Lap noises
    NoisyhistFeature=histFeature+np.rint(np.random.laplace(0, sensitivity_feat / epsilon_feat,size=NUMFEAT)).astype(np.int32)
    #NoisyhistFeature-=np.min(NoisyhistFeature)
    NoisyhistFeature[NoisyhistFeature<0]=0

    return NoisyhistFeature

def getNoisyFeatureEdge(Edges,Features):
    histFeatureEdge=np.zeros((NUMFEAT,NUMFEAT),dtype=np.int32)

    # Counting
    for i in range(NUMUSER) :
        feat1=Features[i]
        for j in range(len(Edges[i])):
            feat2=Features[Edges[i][j]]
            histFeatureEdge[feat1,feat2]+=1

    # Add discrete Lap noisese
    NoisyhistFeatureEdge=histFeatureEdge+np.rint(np.random.laplace(0, sensitivity_edgefeat / epsilon_edgefeat,size=NUMFEAT)).astype(np.int32)
    #NoisyhistFeatureEdge-=np.min(NoisyhistFeatureEdge)
    NoisyhistFeatureEdge[NoisyhistFeatureEdge<0]=0

    return NoisyhistFeatureEdge

def getFeatureEdge(Edges,Features):
    histFeatureEdge=np.zeros((NUMFEAT,NUMFEAT),dtype=np.int32)

    # Counting
    for i in range(NUMUSER) :
        feat1=Features[i]
        for j in range(len(Edges[i])):
            feat2=Features[Edges[i][j]]
            histFeatureEdge[feat1,feat2]+=1
    return histFeatureEdge

def getTriangle(Edges,inputTringle) :
    NumTriangle1 = 0
    NumTriangle2 = 0

    for i in range(NUMUSER) :
        print("Count1: U:" + str(i)+" Now:"+str(NumTriangle1)+" Target:"+str(inputTringle[0]))
        for j in Edges[i] :
            if j < i :
                for k in Edges[j] :
                    if k < i and i in Edges[k] :
                        NumTriangle1 += 1




    for i in range(NUMUSER) :
        print("Count2: U:" + str(i)+" Now:"+str(NumTriangle2)+" Target:"+str(inputTringle[1]))
        combinations = list(itertools.combinations(Edges[i], 2))
        for combo in combinations :
            if combo[0] in Edges[combo[1]] :
                NumTriangle2 += 1
            if combo[1] in Edges[combo[0]] :
                NumTriangle2 += 1

    return [NumTriangle1, NumTriangle2]

def getNoisyTriangle(Edges):
    NumTriangle1=0
    NumTriangle2=0



    for i in range(NUMUSER) :
        print("Count1: "+str(i)+" "+str(NumTriangle1))
        for j in Edges[i]:
            if j <i:
                for k in Edges[j]:
                    if k<i and i in Edges[k] :
                        NumTriangle1 += 1
    print(NumTriangle1)
    NumTriangle1+=np.rint(np.random.laplace(0, sensitivity_triangle / epsilon_triangle)).astype(np.int32)


    for i in range(NUMUSER):
        print("Count2: "+str(i)+" "+str(NumTriangle2))
        combinations = list(itertools.combinations(Edges[i], 2))
        for combo in combinations:
            if combo[0] in Edges[combo[1]]:
                NumTriangle2+=1
            if combo[1] in Edges[combo[0]]:
                NumTriangle2+= 1

    print(NumTriangle2)
    NumTriangle2+=np.rint(np.random.laplace(0, 2*sensitivity_triangle / epsilon_triangle)).astype(np.int32)


    return [NumTriangle1,NumTriangle2]

def getNumEdges(Edges):
    num=0

    for i in range(NUMUSER):
        num+=len(Edges[i])

    return num+np.rint(np.random.laplace(0, sensitivity_numedge / epsilon_numedge)).astype(np.int32)

def getFeatures(protFeature):
    return np.random.choice(np.arange(len(protFeature)), size=NUMUSER, p=protFeature)

def directedTriCycle(histDegree,Triangle,numEdge,Features,A,flag):


    Edges,degreeSeq=directedFCL(histDegree,numEdge)
    #NumTriangle = getTriangle(Edges, Triangle)

    NumTriangle=Triangle-10


    #!!Is to use the given degree, rather than the degree of the generated graph
    outDegreeSeq = np.zeros(np.sum(degreeSeq[:, 0]), dtype=np.int32)
    inDegreeSeq = np.zeros(np.sum(degreeSeq[:, 1]), dtype=np.int32)
    checkedUsr = 0
    for i in range(NUMUSER) :
        num = degreeSeq[i, 0]
        outDegreeSeq[checkedUsr :checkedUsr + num] = i
        checkedUsr += num
    checkedUsr = 0
    for i in range(NUMUSER) :
        num = degreeSeq[i, 1]
        inDegreeSeq[checkedUsr :checkedUsr + num] = i
        checkedUsr += num

    np.random.shuffle(outDegreeSeq)
    np.random.shuffle(inDegreeSeq)


    Qedge = Queue()
    for i in range(NUMUSER):
        for j in Edges[i]:
            Qedge.put([i,j])


    Round=0
    while NumTriangle[0]<Triangle[0] or NumTriangle[1]<Triangle[1]:
        print(Round,":",NumTriangle,Triangle)
        Round+=1

        while 1:
            v_i = np.random.choice(outDegreeSeq)
            if len(Edges[v_i]) == 0:
                continue
            v_k = np.random.choice(Edges[v_i])
            if len(Edges[v_k]) == 0 :
                continue
            else:
                break

        v_j=np.random.choice(Edges[v_k])
        for i in Edges[v_k]:
            if len(Edges[i])>len(Edges[v_j]):
                v_j=i

        if v_i not in Edges[v_j] and v_i!=v_j and (flag or random.random()<=A[Features[v_j]*NUMFEAT+Features[v_i]]) :

            v_q,v_r=Qedge.get()  #Get the oldest edge

            # "Temproally" delete the oldest edge
            Edges[v_q] = [x for x in Edges[v_q] if x != v_r]


            #Count the triangle1s formed by the oldest edge
            Tringle1_oldEdge=0
            for ne in Edges[v_r]:
                if v_q in Edges[ne]:
                    Tringle1_oldEdge+=1


            # Count the triangle1s formed by the new edge
            Tringle1_newEdge = 0
            for ne in Edges[v_i] :
                if v_j in Edges[ne] :
                    Tringle1_newEdge += 1


            if Tringle1_newEdge >= Tringle1_oldEdge:

                # Count the triangle2s formed by the oldest edge
                Tringle2_oldEdge = 0
                for ne in Edges[v_r] :
                    if ne in Edges[v_q] :
                        Tringle2_oldEdge += 1
                for i in range(NUMUSER) :
                    if v_q in Edges[i] and v_r in Edges[i] :
                        Tringle2_oldEdge += 1
                for ne in Edges[v_q] :
                    if v_r in Edges[ne] :
                        Tringle2_oldEdge += 1


                # Count the triangle2s formed by the new edge
                Tringle2_newEdge = 0
                for ne in Edges[v_i] :
                    if ne in Edges[v_j] :
                        Tringle2_newEdge += 1
                for i in range(NUMUSER) :
                    if v_j in Edges[i] and v_i in Edges[i] :
                        Tringle2_newEdge += 1
                for ne in Edges[v_j] :
                    if v_i in Edges[ne] :
                        Tringle2_newEdge += 1


                if Tringle2_newEdge>=Tringle2_oldEdge:
                    #Add the new edge
                    Edges[v_j].append(v_i)
                    Qedge.put([v_j, v_i])
                    NumTriangle[0]+=Tringle1_newEdge-Tringle1_oldEdge
                    NumTriangle[1]+=Tringle2_newEdge-Tringle2_oldEdge
                else:
                    Qedge.put([v_q, v_r])

            else:
                Qedge.put([v_q, v_r])


        while 1:
            v_i = np.random.choice(outDegreeSeq)
            if len(Edges[v_i]) == 0:
                continue
            v_k = np.random.choice(Edges[v_i])
            if len(Edges[v_k]) == 0 :
                continue
            else:
                break

        v_j = np.random.choice(Edges[v_k])

        for i in Edges[v_k]:
            if len(Edges[i])>len(Edges[v_j]):
                v_j=i



        if v_j not in Edges[v_i] and v_i!=v_j and (flag or random.random()<=A[Features[v_j]*NUMFEAT+Features[v_i]]):

            v_q, v_r = Qedge.get()  # Get the oldest edge

            # "Temproally" delete the oldest edge
            Edges[v_q] = [x for x in Edges[v_q] if x != v_r]


            # Count the triangle1s formed by the oldest edge
            Tringle1_oldEdge = 0
            for ne in Edges[v_r] :
                if v_q in Edges[ne] :
                    Tringle1_oldEdge += 1


            # Count the triangle1s formed by the new edge
            Tringle1_newEdge = 0
            for ne in Edges[v_j] :
                if v_i in Edges[ne] :
                    Tringle1_newEdge += 1


            if Tringle1_newEdge >= Tringle1_oldEdge:

                # Count the triangle2s formed by the oldest edge
                Tringle2_oldEdge = 0
                for ne in Edges[v_r] :
                    if ne in Edges[v_q] :
                        Tringle2_oldEdge += 1
                for i in range(NUMUSER) :
                    if v_q in Edges[i] and v_r in Edges[i] :
                        Tringle2_oldEdge += 1
                for ne in Edges[v_q] :
                    if v_r in Edges[ne] :
                        Tringle2_oldEdge += 1


                # Count the triangle2s formed by the new edge
                Tringle2_newEdge = 0
                for ne in Edges[v_j] :
                    if ne in Edges[v_i] :
                        Tringle2_newEdge += 1
                for i in range(NUMUSER) :
                    if v_i in Edges[i] and v_j in Edges[i] :
                        Tringle2_newEdge += 1
                for ne in Edges[v_i] :
                    if v_j in Edges[ne] :
                        Tringle2_newEdge += 1

                if Tringle2_newEdge >= Tringle2_oldEdge:
                    # Add the new edge
                    Edges[v_i].append(v_j)
                    Qedge.put([v_i, v_j])
                    NumTriangle[0] += Tringle1_newEdge - Tringle1_oldEdge
                    NumTriangle[1] += Tringle2_newEdge - Tringle2_oldEdge
                else :
                    Qedge.put([v_q, v_r])
            else:
                Qedge.put([v_q, v_r])




    return Edges

def getTriangle1(Edges,edge):

    Num= 0
    v_q,v_r=edge
    for ne in Edges[v_r] :
        if v_q in Edges[ne] :
            Num += 1

    return Num

def getTriangle2(Edges, edge) :
    Num = 0
    v_i,v_j=edge
    for ne in Edges[v_j] :
        if ne in Edges[v_i] :
            Num += 1
    for i in range(NUMUSER) :
        if v_i in Edges[i] and v_j in Edges[i] :
            Num += 1
    for ne in Edges[v_i] :
        if v_j in Edges[ne] :
            Num += 1

    return Num

def directedFCL(histDegree,numEdge):

    checkedUsr = 0
    degreeSeq=np.zeros((NUMUSER,2),dtype=np.int32)

    for x in range(1,NUMUSER):
        i=x
        for j in range(x+1):
            num = histDegree[i, j]
            histDegree[i, j]=0
            if num <= 0 :
                continue
            if checkedUsr + num > NUMUSER :
                num = NUMUSER - checkedUsr + 1
            degreeSeq[checkedUsr :checkedUsr + num, :] = [i, j]
            checkedUsr += num
            if checkedUsr >= NUMUSER :
                break
        if checkedUsr >= NUMUSER :
            break
        j=x
        for i in range(x):
            num = histDegree[i, j]
            histDegree[i, j] = 0
            if num <= 0 :
                continue
            if checkedUsr + num > NUMUSER :
                num = NUMUSER - checkedUsr + 1
            degreeSeq[checkedUsr :checkedUsr + num, :] = [i, j]
            checkedUsr += num
            if checkedUsr >= NUMUSER :
                break
        if checkedUsr >= NUMUSER :
            break

    #Initianlzie the edge list #to make sure that the graph is connected:
    Edges={}
    for i in range(NUMUSER):
        Edges[i]=[]



    #Contructing the sampling vectors:
    outDegreeSeq=np.zeros(np.sum(degreeSeq[:,0]),dtype=np.int32)
    inDegreeSeq=np.zeros(np.sum(degreeSeq[:,1]),dtype=np.int32)


    checkedUsr = 0
    for i in range(NUMUSER):
        num = degreeSeq[i, 0]
        outDegreeSeq[checkedUsr :checkedUsr + num] = i
        checkedUsr += num

    checkedUsr = 0
    for i in range(NUMUSER) :
        num = degreeSeq[i, 1]
        inDegreeSeq[checkedUsr :checkedUsr + num] = i
        checkedUsr += num

    np.random.shuffle(outDegreeSeq)
    np.random.shuffle(inDegreeSeq)


    #Fast CL model (directed version)
    #Rnadomly sampling nodes:
    np.random.shuffle(outDegreeSeq)
    np.random.shuffle(inDegreeSeq)
    sampled_nodes1 = np.random.choice(outDegreeSeq, size=numEdge-NUMUSER)
    sampled_nodes2 = np.random.choice(inDegreeSeq, size=numEdge-NUMUSER)


    for i in range(numEdge-NUMUSER):
        Edges[sampled_nodes1[i]].append(sampled_nodes2[i])



    #Delete repeated elemetns from Edges
    numDelEdges=0
    for i in range(NUMUSER):
        unique_elements = np.unique(Edges[i])
        numDelEdges+=len(Edges[i]) - len(unique_elements)

        if i in unique_elements:    #Delete self-contained edge
            numDelEdges+=1
            contains_element = np.in1d(unique_elements, i)
            Edges[i] = unique_elements[~contains_element].tolist()
        else:
            Edges[i] = unique_elements.tolist()

    #Padding deleted edges:
    for i in range(numDelEdges):
        while 1:
            node1 = np.random.choice(outDegreeSeq)
            node2 = np.random.choice(inDegreeSeq)
            if node2 not in Edges[node1] and node2!=node1:
                Edges[node1].append(node2)
                break



    """
    #Padding edges to isolated vertices (这一步可以放在生成图的最后)
    inDegrees = np.zeros(NUMUSER, dtype=np.int32)
    outDegrees = np.zeros(NUMUSER, dtype=np.int32)
    for i in range(NUMUSER) :
        edge = Edges[i]
        for j in range(len(edge)) :
            inDegrees[edge[j]] += 1
    for i in range(NUMUSER) :
        outDegrees[i] = len(Edges[i])
    for i in range(NUMUSER):
        if inDegrees[i] == 0 and outDegrees[i]==0: #Vertex i is a isolated vertex
            node1 = np.random.choice(outDegreeSeq)  # Randomly select vertex
            node2 = np.random.choice(inDegreeSeq)  # Randomly select vertex

            if random.choice([0, 1]): #Randomly decide add an outedge or a in-edge to vertex i
                Edges[node1].append(i)
            else:
                Edges[i].append(node2)
    """


    return Edges,degreeSeq

def calculate_cluster_coefficient(Edges):
    total_coefficient = 0.0
    node_count = len(Edges)

    for node in graph:
        neighbors = Edges[node]
        actual_edges = 0
        possible_edges = len(neighbors) * (len(neighbors) - 1)

        for neighbor in neighbors:
            if neighbor in Edges and node in graph[neighbor]:
                actual_edges += 1

        if possible_edges > 0:
            node_coefficient = (2.0 * actual_edges) / possible_edges
            total_coefficient += node_coefficient

    if node_count > 0:
        average_coefficient = total_coefficient / node_count
        return average_coefficient
    else:
        return 0.0

def extractGraph(flag):


    Edges, Features = fileIO.graphIO(flag)

    NoisyhistDegree,histDegree=getNoisyhistDegree(Edges)

    NoisyhistFeature=getFeature(Features)

    NoisyhistFeatureEdge=getNoisyFeatureEdge(Edges,Features)

    NoisyTriangle=getNoisyTriangle(Edges)

    NoisynumEdges=getNumEdges(Edges)

    #Writing the extracted features to the .txt file
    if flag == 1 :   #twitter dataset
        directory = "GraphFeatures/twitter/" # file path

    np.save(directory+"histDegree1.npy",NoisyhistDegree)
    np.save(directory+"OriginalhistDegree1.npy",histDegree)
    np.save(directory+"histFeature1.npy",NoisyhistFeature)
    np.save(directory+"histFeatureEdge1.npy",NoisyhistFeatureEdge)
    np.save(directory+"Triangle1.npy",NoisyTriangle)
    np.save(directory+"NoisynumEdges1.npy",NoisynumEdges)




def genGraph(flag):
    if flag == 1 :   #twitter dataset
        directory = "GraphFeatures/twitter/" # file path

    #Read graph features:
    histDegree=np.load(directory+"histDegree1.npy")
    histDegree[0,0]=0
    histFeature=np.load(directory+"histFeature1.npy")
    histFeatureEdge=np.load(directory+"histFeatureEdge1.npy")
    numTriangle=np.load(directory+"Triangle1.npy")
    numEdge=np.load(directory+"NoisynumEdges1.npy")


    #Normalization:
    protFeature=histFeature/np.sum(histFeature)
    protFeatureEdge=histFeatureEdge/np.sum(histFeatureEdge)


    Y_F=np.zeros((NUMFEAT*NUMFEAT,2),dtype=np.int32)  #Set of elements representing possible edge attribute configurations
    A=np.zeros(NUMFEAT*NUMFEAT,dtype=np.float)
    A_old=np.ones(NUMFEAT*NUMFEAT,dtype=np.float)
    for i in range(NUMFEAT):
        for j in range(NUMFEAT):
            Y_F[i*NUMFEAT+j,:]=[i,j]

    #Generate graph:
    Features=getFeatures(protFeature)   #Sample new attribute vectors

    Edges=directedTriCycle(histDegree,numTriangle,numEdge,Features,A,True)

    R=np.zeros(NUMFEAT*NUMFEAT,dtype=np.float)
    flag=0
    while 1:

        FeatureEdge_=getFeatureEdge(Edges, Features)
        proFeatureEdge_=FeatureEdge_/np.sum(FeatureEdge_)
        for i in range(NUMFEAT*NUMFEAT):
            f1,f2=Y_F[i,:]
            if proFeatureEdge_[f1,f2]!=0:
                R[i]=protFeatureEdge[f1,f2]/proFeatureEdge_[f1,f2]
            if flag:
                R[i]=R[i]*A_old[i]
        for i in range(NUMFEAT*NUMFEAT):
            A[i]=R[i]/np.max(R)

        Edges=directedTriCycle(histDegree,numTriangle,numEdge,Features,A,False)
        print(hellinger_distance(A_old,A))
        if hellinger_distance(A_old,A)<0.01:
            return Edges, Features
        A_old=A
        flag=1