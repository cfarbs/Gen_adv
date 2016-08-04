#!/usr/bin/env python

##############################################
# A combined runner program to run a generative adversarial network
# to generate novel protein sequences.
# This implementation generates portions of helices based on real data.
##############################################

################## DEBUG TOGGLE #################

Debug = False

#################################################

#Generator packages
import sys
import os
import time
import numpy as np
from GRU.gen_utils import *
from datetime import datetime, timedelta
from GRU.gru_theano import GRUTheano
import time

#Discriminator packages
from helix_gen_util import main, parse_args

#Repeat packages
from gen_assembler import Positives, Num2Prot


#################### INITIAL GENERATOR CODE ###############################
#Timers for debugging to determine when certain events happen
start_init = time.monotonic()


start_GRU_train = time.monotonic()
start_GRU = time.monotonic()
start_overall = time.monotonic()

#Assign all variables required for the generator.
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "24"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "GRU/Init_seq.pkl")
#Because adversarial is false, the network will look in the above data file for data.
ADVERSARIAL = os.environ.get("ADVERSARIAL", False)

#Names the generator file based on the time and day it was made
if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "gen_models/GRU-%s-%s-%s-%s-initial.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

#Load the initial dataset from the pickle file, tokenize it, and prepare it for use as training data
x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE, ADVERSARIAL)

#Creates an initial GRU model
init_model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

# Print SGD step time
t1 = time.time()
init_model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print ("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
sys.stdout.flush()

#Trains the network with sgd
for epoch in range(NEPOCH):
  train_with_sgd(init_model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
    callback_every=PRINT_EVERY, callback=sgd_callback)

#Once training is complete, prints how much time has elapsed.
end_gru_train = time.monotonic()
elapsed_gru_train = timedelta(seconds=end_gru_train - start_GRU_train)
print ("GRU network training complete!")
print ("Time elapsed: %s" % (elapsed_gru_train))

#Generate a data set for the discriminator to look at
init_data = generate_sentences(init_model, 1000, index_to_word, word_to_index)

#Take the overall time required for the initial training.
end_gru = time.monotonic()
elapsed_gru = timedelta(seconds=end_gru - start_GRU)
print ("Generative Network initialization COMPLETE.")
print ("Time elapsed for GRU initial training: %s"%(elapsed_gru))
######################## INITIAL DISCRIMINATOR CODE ########################
#Timer to determine how long discriminator takes to discriminate
start_disc = time.monotonic()

args = parse_args()
errors, probs = main(args)

#List to contain the means of errors for each run
errorlist = []
errorlist.append(errors)

#Prints time elapsed for discriminator step and initialization step
end_disc = time.monotonic()
#end_init = time.monotonic()
elapsed_disc = timedelta(seconds=end_disc - start_disc)
#elapsed_init = timedelta(seconds=end_init - start_init)
print ("Time elapsed for Discriminator initial training: %s"%(elapsed_disc))
#print ("Time elapsed for initial discriminative run: %s"%(elapsed_init))

####################### REPEATED SEGMENT - GENERATIVE ADVERSARIAL ###################
#Starts a timer to track how long repeated segment takes

start_repeat = time.monotonic()

#Generator now will use data points from the discriminator to train.

###################### FOR DEBUGGING ##############################
if Debug:
    #Assign all variables required for the generator.
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
    VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "24"))
    EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
    HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
    NEPOCH = int(os.environ.get("NEPOCH", "20"))
    MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
    PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))
    INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "GRU/Init_seq.pkl")
######################################################################

ADV = os.environ.get("ADV", True)

#Resets the initial data variable so it can be used in the repeated segment.
init_data = []

#List to store the true positives found by the generator, and feed them to the discriminator
repeat_data = Positives(probs)
repeat_data = Num2Prot(repeat_data)
print (type(repeat_data))
print (repeat_data[0])
#Variable to count how many iterations it takes for the generator to fool the discriminator
iternum = 1

while np.mean(errorlist) not in range(48,53):
    #Repeated generator code; trains with repeat_data
    if not MODEL_OUTPUT_FILE:
      ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
      MODEL_OUTPUT_FILE = "gen_models/GRU-%s-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM, iternum)

    x_train, y_train, word_to_index, index_to_word = load_data(ADV, repeat_data, VOCABULARY_SIZE)

    model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

    # Print SGD step time
    t1 = time.time()
    model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
    t2 = time.time()
    print ("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
    sys.stdout.flush()

    for epoch in range(NEPOCH):
      train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
        callback_every=PRINT_EVERY, callback=sgd_callback)
    #Creates another dataset for the discriminator
    init_data = generate_sentences(model, 1000, index_to_word, word_to_index)

    #Discriminator discriminates, using same arguments passed initially.
    errors, probs = classify_with_network2(**nn_args)
    new_data = Positives(probs)
    #New true positives are appended to training corpus
    new_data = Num2Prot(new_data)
    repeat_data.append(new_data)
    #New error rate appended to list of errors
    errorlist.append(errors)
    #Appends the number of iterations
    iternum += 1

#Takes the final total elapsed time
end_repeat = time.monotonic()
end_overall = time.monotonic()
elapsed_repeat = timedelta(seconds=end_repeat - start_repeat)
elapsed_overall = timedelta(seconds=end_overall - start_overall)

#prints and summarizes the run
print ("Generative Adversarial Network complete!")
print ("Number of iterations to train: %s"%(iternum))
print ("Time spent iterating: %s" % (elapsed_repeat))
print ("Final test accuracy: %s"%(errorlist[len(errorlist)-1]))
print ("Overall time elapsed: %s"%(elapsed_overall))
