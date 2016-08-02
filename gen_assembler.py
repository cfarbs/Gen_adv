#!/usr/bin/env python

import numpy as np

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
            print (prob[1])
            print ("Some how, a third value was decided on by a binary classifier. Weird, huh?")
    return positives
