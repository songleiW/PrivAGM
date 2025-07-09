import random
import numpy as np
import sys
import threading
import fileIO
np.set_printoptions(threshold=np.inf)

NUMUSER=76244
p=0.05


def DataCollection(flag): #Data collecton phase


    Edges, Features = fileIO.graphIO(flag)

    if flag == 1 :  # twitter dataset;
        directory = "CiphertextTwitter/"  # file path


    #Encryt the fectures
    FeaturesP1=np.random.randint(np.iinfo(np.int32).max, size=(2,NUMUSER),dtype=np.int32)
    FeaturesP2=FeaturesP1.copy()
    FeaturesP2[0,:]=Features-FeaturesP1[0,:]

    FeaturesP3=FeaturesP2.copy()
    FeaturesP3[0,:]=Features-FeaturesP2[1,:]

    FeaturesP3[1,:]=Features-FeaturesP1[1,:]

    #Save the ciphertext of features
    np.save(directory + "P1_feature.npy", FeaturesP1)
    np.save(directory + "P2_feature.npy", FeaturesP2)
    np.save(directory + "P3_feature.npy", FeaturesP3)

    EdgesP1 = {}
    EdgesP2 = {}
    EdgesP3 = {}

    #Encryt the edges
    encEdge = [[] for _ in range(3)]
    for i in range(NUMUSER):
        print(i)
        encEdge[0].clear()
        encEdge[1].clear()
        encEdge[2].clear()

        for ID in range(NUMUSER):
            if ID in Edges[i]:  #The edge is true
                selectedID=random.sample([1,2,3], 2)
                e1=[ID,np.random.randint(np.iinfo(np.int32).max),selectedID[1]]
                e2=[ID,1-e1[1],selectedID[0]]

                encEdge[selectedID[0]-1].append(e1)
                encEdge[selectedID[1]-1].append(e2)
            else:   #The edge is false
                if random.random()<p:   #Add dummy edge
                    selectedID = random.sample([1, 2, 3], 2)
                    e1 = [ID, np.random.randint(np.iinfo(np.int32).max), selectedID[1]]
                    e2 = [ID, 0 - e1[1], selectedID[0]]

                    encEdge[selectedID[0] - 1].append(e1)
                    encEdge[selectedID[1] - 1].append(e2)


        EdgesP1[i]=encEdge[0].copy()
        EdgesP2[i]=encEdge[1].copy()
        EdgesP3[i]=encEdge[2].copy()

    # Save the ciphertext of features
    np.save(directory + "P1_edges.npy", EdgesP1)
    np.save(directory + "P2_edges.npy", EdgesP2)
    np.save(directory + "P3_edges.npy", EdgesP3)

    return 0












