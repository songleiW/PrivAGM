import numpy as np
import glob
import os
NUMFATURE=10
NUMUSER=76244

def dataPreprocess(flag) :   #Datasets preprocessing
    #Read files:

    if flag == 1 :   #twitter dataset; 76244 vertices; 38 features;
        directory = "Datasets/twitter" # file path
        edge_file_pattern = os.path.join(directory,f'*.{"edges"}')  # Construct the file matching pattern for edges
        edge_file_list = glob.glob(edge_file_pattern) # Get a list of matching file path for edges

        feat_file_pattern = os.path.join(directory, f'*.{"feat"}')  # Construct the file matching pattern for features
        feat_file_list = glob.glob(feat_file_pattern)  # Get a list of matching file path for features


    if flag == 2 :   #gplus dataset; 107614 vertices; 38 features;
        directory = "Datasets/gplus" # file path
        edge_file_pattern = os.path.join(directory,f'*.{"edges"}')  # Construct the file matching pattern for edges
        edge_file_list = glob.glob(edge_file_pattern) # Get a list of matching file path for edges

        feat_file_pattern = os.path.join(directory, f'*.{"feat"}')  # Construct the file matching pattern for features
        feat_file_list = glob.glob(feat_file_pattern)  # Get a list of matching file path for features


    Edeges={}
    IDmappings={}
    for file_path in edge_file_list:
        with open(file_path, 'r') as file :
            # Read the edges
            line = file.readline().strip()
            while line :
                index = line.split(' ')

                if index[0]==index[1]:  # Self-connection is not considered
                    line = file.readline().strip()
                    continue

                if index[0] not in IDmappings:
                    IDmappings[index[0]]=len(IDmappings)

                if index[1] not in IDmappings:
                    IDmappings[index[1]]=len(IDmappings)


                ID1=IDmappings[index[0]]
                ID2=IDmappings[index[1]]


                if ID1 in Edeges :
                    if ID2 not in Edeges[ID1]:
                        Edeges[ID1].append(ID2)  #append a neighbroing vertex
                else:
                    Edeges[ID1]=[ID2]

                line = file.readline().strip()
        file.close()



    Features={}
    FeatMappings={}
    for file_path in feat_file_list :
        with open(file_path, 'r') as file :
            # Read the features
            line = file.readline().strip()
            while line:
                index = line.split(' ',1)
                if index[0] in IDmappings:
                    ID=IDmappings[index[0]]
                    fecture=index[1][:NUMFATURE]
                    if fecture not in FeatMappings:
                        FeatMappings[fecture]=len(FeatMappings)
                    Features[ID]=FeatMappings[fecture]
                line = file.readline().strip()

        file.close()


    #Write files:
    if flag == 1 :   #twitter dataset
        directory = "Datasets/de_twitter/" # file path
    if flag == 2 :   #gplus dataset
        directory = "Datasets/de_gplus/" # file path


    for key in IDmappings:
        userID=IDmappings[key]
        file = open(directory+str(userID)+".txt", "w")
        file.write(str(userID)+"\n")
        if userID not in Features:
            Features[userID]=0
        file.write(str(Features[userID])+"\n")
        if userID in Edeges:
            for i in range(len(Edeges[userID])):
                file.write(str(Edeges[userID][i]) + "\n")

        file.close()

def dataPreprocessMooc() :   #Datasets preprocessing

    file=open("Datasets/mooc/edges.tsv", 'r')
    Edeges = {}
    IDmappings = {}
    # Read the edges
    line = file.readline().strip()
    while line :
        string = line.split('\t')
        index=string[1:2]

        if index[0] == index[1] :  # Self-connection is not considered
            line = file.readline().strip()
            continue

        if index[0] not in IDmappings :
            IDmappings[index[0]] = len(IDmappings)

        if index[1] not in IDmappings :
            IDmappings[index[1]] = len(IDmappings)

        ID1 = IDmappings[index[0]]
        ID2 = IDmappings[index[1]]

        if ID1 in Edeges :
            if ID2 not in Edeges[ID1] :
                Edeges[ID1].append(ID2)  # append a neighbroing vertex
        else :
            Edeges[ID1] = [ID2]

        line = file.readline().strip()
    file.close()



    Features={}
    FeatMappings={}
    file=open("Datasets/mooc/features.tsv", 'r')
    # Read the features
    line = file.readline().strip()
    while line:
        index = line.split('\t')
        if index[1] in IDmappings:
            ID=IDmappings[index[1]]
            fecture=index[1][:NUMFATURE]
            if fecture not in FeatMappings:
                FeatMappings[fecture]=len(FeatMappings)
            Features[ID]=FeatMappings[fecture]
        line = file.readline().strip()

    file.close()


    #Write files:
    directory = "Datasets/de_mooc/" # file path

    for key in IDmappings:
        userID=IDmappings[key]
        file = open(directory+str(userID)+".txt", "w")
        file.write(str(userID)+"\n")
        file.write(str(Features[userID])+"\n")
        if userID in Edeges:
            for i in range(len(Edeges[userID])):
                file.write(str(Edeges[userID][i]) + "\n")

        file.close()

def graphIO(flag):
    if flag == 1 :   #twitter dataset; 76245 vertices; 38 features;
        directory = "Datasets/de_twitter" # file path
        user_file_pattern = os.path.join(directory,f'*.{"txt"}')  # Construct the file matching pattern for users
        user_file_list = glob.glob(user_file_pattern) # Get a list of matching file path for users

    if flag == 2 :   #gplus dataset; 76245 vertices; 38 features;
        directory = "Datasets/de_gplus" # file path
        user_file_pattern = os.path.join(directory,f'*.{"txt"}')  # Construct the file matching pattern for users
        user_file_list = glob.glob(user_file_pattern) # Get a list of matching file path for users

    if flag == 3 :   #twitter dataset; 76245 vertices; 38 features;
        directory = "Datasets/de_mooc" # file path
        user_file_pattern = os.path.join(directory,f'*.{"txt"}')  # Construct the file matching pattern for users
        user_file_list = glob.glob(user_file_pattern) # Get a list of matching file path for users


    Features=np.zeros(NUMUSER, dtype=int)
    Edges={}
    for file_path in user_file_list:
        with open(file_path, 'r') as file :
            # Read the ID
            line = file.readline().strip()
            num=int(line)
            Edges[num]=[]
            # Read the fecture
            line = file.readline().strip()
            Features[num]=int(line)
            # Read the edges
            line = file.readline().strip()

            while line :
                neigh=int(line)
                Edges[num].append(neigh)
                line = file.readline().strip()

        file.close()


    return Edges,Features