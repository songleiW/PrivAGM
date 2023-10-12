import time
import fileIO
import Phase1
import Phase2
import Plaintext
import numpy as np

NUMUSER=76244
flag=1

#fileIO.dataPreprocess(flag)

start_time = time.time()

#EdgesP1, EdgesP2, EdgesP3, FeaturesP1, FeaturesP2, FeaturesP3=Phase1.DataCollection(flag) #Data collecton phase


#Phase2.secExtractGraph(flag) #Secure extraction phase

Plaintext.extractGraph(flag)  #Extract the features of input graph

Edges,Features=Plaintext.genGraph(flag)    #Generate graph based on the extracted feactures.

end_time = time.time()
run_time = end_time - start_time

print("Running time: ", run_time, "s")




