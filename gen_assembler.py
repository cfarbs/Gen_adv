#!/usr/bin/env python

import numpy as np
from helix_network.lib.prot_to_num import amino_dict


def Positives(data):
    truepos = 0
    trueneg = 0
    falsepos = 0
    falseneg = 0
    positives = []
    for prob in data:
        if prob[1] == 0:
            if prob[2][0] > prob[2][1]:
                truepos +=1
            else:
                falseneg += 1
        elif prob[1] == 1:
            if prob[2][1] > prob [2][0]:
                trueneg += 1
            else:
                falsepos +=1
                positives.append(prob[0])
        else:
            print ("Some how, a third value was decided on by a binary classifier. Weird, huh?")

    return positives

#Module to convert amino acid in digit form back into numbers
def Num2Prot(data):
    digitseq = []
    amino = amino_dict()
    amino2num = {}
    for acids in amino.keys():
        amino2num[str(amino[acids])]=acids
    for count, aa in enumerate(data):
        tempdigi = []
        for residue in range(len(aa)):
            tempdigi.append(str(amino2num[str(aa[residue])]))
            if len(tempdigi) == len(aa):
                digitseq.append(tempdigi)
    return digitseq

def Value2Key(data,dictionary):
    valuedict = {}
    digitseq = []
    keylength = 0
    for keys in dictionary.keys():
        valuedict[str(dictionary[keys])]=keys
    print (valuedict.keys())
    for keys in valuedict.keys():
        string = keys.strip("[")
        string = string.strip("]")
        string = string.split(" ")
        keylength = len(string)
    tempdigi = []
    for count, aa in enumerate(data):
        reslist = []
        for residue in aa:
            #print (residue)
            residue = str(residue)
            reslist.append(residue)
            if len(reslist)==keylength:
                tempfeats = ", ".join(reslist)
                tempfeats = "["+tempfeats+"]"
                tempfeats = str(tempfeats)
                #print (valuedict[tempfeats])
                #print (tempfeats)
                if tempfeats in valuedict.keys():
                    #print ("Found one!")
                    #print (tempfeats)
                    tempdigi.append(str(valuedict[tempfeats]))
                    #print (tempdigi)
                    #print(len(tempdigi))
                    if len(tempdigi) == len(aa)/keylength:
                        print (tempdigi)
                        digitseq.append(tempdigi)
                        reslist = []
                        tempdigi = []

    return digitseq
