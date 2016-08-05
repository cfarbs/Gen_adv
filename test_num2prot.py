#!/usr/bin/env python
import pickle
from gen_assembler import Num2Prot
from random import shuffle, sample

"""outsent = pickle.load(open("helix_network/lib/gen_helices.pkl","rb"))
#print (len(outsent))
#data = Num2Prot(outsent)
#print(data[30])
test = outsent
print (len(test))

short = [1,5,3,6,2,6,1,11,9,10]
print (len(short))
if len(test) != len(short):
    test_short = sample(test,len(short))
print (len(test_short))
print (test_short)"""

filename = "GRU/Init_seq.pkl"

f = pickle.load(open(filename,"rb"))
print (f)
