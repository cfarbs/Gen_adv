#!/usr/bin/env python

#Generator packages
import sys
import os
import time
import numpy as np
from GRU.gen_utils import *
from datetime import datetime
from GRU.gru_theano import GRUTheano

#Discriminator packages
import sys
import pickle
from helix_network.lib.helix_neural_network import classify_with_network2
from argparse import ArgumentParser

#################### INITIAL GENERATOR CODE ###############################

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "24"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))
#Need to figure out how I'm going to start with one infile and change to another
#INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "rawhelices.pkl")

if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s-initial.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

init_model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

# Print SGD step time
t1 = time.time()
init_model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print ("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
sys.stdout.flush()

for epoch in range(NEPOCH):
  train_with_sgd(init_model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
    callback_every=PRINT_EVERY, callback=sgd_callback)

init_data = generate_sentences(init_model, 1000, index_to_word, word_to_index)



######################## INITIAL DISCRIMINATOR CODE ########################
