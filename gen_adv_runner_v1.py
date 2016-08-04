#!/usr/bin/env python

##############################################
# A combined runner program to run a generative adversarial network
# to generate novel protein sequences.
# This implementation generates portions of helices based on real data.
##############################################

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
import sys
import pickle
from helix_network.lib.helix_neural_network import classify_with_network2
from argparse import ArgumentParser
#from multiprocessing import Process, current_process, Manager

#Repeat packages
from gen_assembler import Positives, Num2Prot


#################### INITIAL GENERATOR CODE ###############################
#Timers for debugging to determine when certain events happen
start_init = time.monotonic()
"""start_GRU_train = time.monotonic()
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
print ("Time elapsed for GRU initial training: %s"%(elapsed_gru))"""
######################## INITIAL DISCRIMINATOR CODE ########################
#Timer to determine how long discriminator takes to discriminate
start_disc = time.monotonic()

#Lists initialized to store the error rate and test data points.
errors = []
probs = []

#An argument parser to introduce arguments into the discriminator.
#The network starts adversarial, as it will be taking data from live memory,
#rather than from a file.
def parse_args():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument('--config_file', '-c', action='store', type=str, dest='config',
                        required=True, help='config file (pickle)')
    parser.add_argument('--model_dir', action='store', type=str, dest='model_file', required=False,
                        default=None, help="directory with models")
    parser.add_argument('--jobs', '-j', action='store', dest='jobs', required=False,
                        default=4, type=int, help="number of jobs to run concurrently")
    parser.add_argument('--iter', '-i', action='store', dest='iter', required=False,
                        default=1, type=int, help="number of iterations to do")
    parser.add_argument('--learning_algorithm', '-a', dest='learning_algo', required=False,
                        default=None, action='store', type=str, help="options: \"annealing\"")
    parser.add_argument('--epochs', '-ep', action='store', dest='epochs', required=False,
                        default=10000, type=int, help="number of iterations to do")
    parser.add_argument('--batch_size', '-b', action='store', dest='batch_size', required=False, type=int,
                        default=None, help='specify batch size')
    parser.add_argument('--learning_rate', '-e', action='store', dest='learning_rate',
                        required=False, default=0.01, type=float)
    parser.add_argument('--L1_reg', '-L1', action='store', dest='L1', required=False,
                        default=0.0, type=float)
    parser.add_argument('--L2_reg', '-L2', action='store', dest='L2', required=False,
                        default=0.001, type=float)
    parser.add_argument('--train_test', '-s', action='store', dest='split', required=False,
                        default=0.9, type=float, help="train/test split")
    parser.add_argument('--preprocess', '-p', action='store', required=False, default=None,
                        dest='preprocess', help="options:\nnormalize\ncenter\ndefault:None")
    parser.add_argument('--output_location', '-o', action='store', dest='out',
                        required=True, type=str, default=None,
                        help="directory to put results")
    parser.add_argument('--adversarial', '-ad', action='store', dest='adversarial',
                        required=True, type=str, default=True,
                        help="determines whether run will be adversarial")

    args = parser.parse_args()
    return args

#Code to run the discriminator as a multiprocessing function
#def run_nn2(work_queue, done_queue):
#    try:
#        for f in iter(work_queue.get, 'STOP'):
#            n = classify_with_network2(**f)
#    except Exception:
#        done_queue.put("%s failed" % current_process().name)

def main(args):
    args = parse_args()

    #Config files are made with make_helix_config.py
    #Every config file is a pickled dictionary.
    config = pickle.load(open(args.config, 'rb'))

    #Only certain arguments are sourced from the config file.
    #Additional arguments may be added under extra_args.
    try:
            extra_args = config['extra_args']
            batch_size = extra_args['batch_size']
    except KeyError:
            extra_args = None
            batch_size = args.batch_size

    #Whether from the config file or the command line, there must always be a batch size.
    assert(batch_size is not None), "You need to specify batch_size with a flag or have it in the config file"

#Prints out a start message, describing the arguments and configurations used for the run.
    start_message = """
#    Starting Neural Net analysis for {title}
#    Command line: {cmd}
#    Config file: {config}
#    Network type: {type}
#    Network dims: {dims}
#    Adversarial: {adv}
#    Importing models from {models}
#    Learning algorithm: {algo}
#    Batch size: {batch}
#    Iterations: {iter}.
#    Epochs: {epochs}
#    Data pre-processing: {center}
#    Train/test split: {train_test}
#    L1 reg: {L1}
#    L2 reg: {L2}
#    Output to: {out}""".format(iter=args.iter,
                                train_test=args.split, out=args.out, epochs=args.epochs, center=args.preprocess,
                                L1=args.L1, L2=args.L2, type=config['model_type'], dims=config['hidden_dim'],
                                cmd=" ".join(sys.argv[:]), title=config["experiment_name"],
                                batch=batch_size, algo=args.learning_algo, models=args.model_file,
                                config=args.config, adv=args.adversarial)
#
#
    print (sys.stdout, start_message)

# More code related to multiprocessing jobs
#    workers = args.jobs
#    work_queue = Manager().Queue()
#    done_queue = Manager().Queue()
#    jobs = []

######## FOR DEBUGGING; MAKE SURE TO REMOVE! ###################
    init_data = pickle.load(open("helix_network/lib/gen_helices.pkl","rb"))
    adversarial = args.adversarial
################################################################

#Loads the network to run experiments equal to the number prescribed by the dictionary.
    for experiment in range(len(config['helixdict'])):
        if adversarial:
            nn_args = {
                "preprocess": args.preprocess,
                "title": config["helixdict"][experiment]['title'],
                "learning_algorithm": args.learning_algo,
                "train_test_split": args.split,
                "iterations": args.iter,
                "epochs": args.epochs,
                "batch_size": batch_size,
                "learning_rate": args.learning_rate,
                "L1_reg": args.L1,
                "L2_reg": args.L2,
                "hidden_dim": config['hidden_dim'],
                "model_type": config['model_type'],
                "model_dir": args.model_file,
                "extra_args": extra_args,
                "out_path": args.out,
                "helixdict": config['helixdict'][experiment],
                "adversarial": args.adversarial,
                "data": init_data
            }
        else:
            nn_args = {
                "preprocess": args.preprocess,
                "title": config["helixdict"][experiment]['title'],
                "learning_algorithm": args.learning_algo,
                "train_test_split": args.split,
                "iterations": args.iter,
                "epochs": args.epochs,
                "batch_size": batch_size,
                "learning_rate": args.learning_rate,
                "L1_reg": args.L1,
                "L2_reg": args.L2,
                "hidden_dim": config['hidden_dim'],
                "model_type": config['model_type'],
                "model_dir": args.model_file,
                "extra_args": extra_args,
                "out_path": args.out,
                "helixdict": config['helixdict'][experiment],
                "adversarial": args.adversarial,
                "data": None
            }
        #Activate for debugging, but also if a multiprocess run is not desired
        errors, probs = classify_with_network2(**nn_args)  # activate for debugging

#The commented code between lines 217 and 235 is related to multiprocessing.
#        work_queue.put(nn_args)
#        print (probs[5])
#    for w in range(workers):
        #if args.group_3 is None:
#        p = Process(target=run_nn2, args=(work_queue, done_queue))
        #else:
            #p = Process(target=run_nn3, args=(work_queue, done_queue))
#        p.start()
#        jobs.append(p)
#        work_queue.put('STOP')


#    for p in jobs:
#        p.join()

#    done_queue.put('STOP')
#Prints a finished statement once a run is complete
    print (sys.stderr, "\n\tFinished Neural Net")
#Prints time elapsed for discriminator step and initialization step
    end_disc = time.monotonic()
    end_init = time.monotonic()
    elapsed_disc = timedelta(seconds=end_disc - start_disc)
    elapsed_init = timedelta(seconds=end_init - start_init)
    print ("Time elapsed for Discriminator initial training: %s"%(elapsed_disc))
    print ("Time elapsed for initial discriminative run: %s"%(elapsed))

if __name__ == "__main__":
    sys.exit(main(sys.argv))

####################### REPEATED SEGMENT - GENERATIVE ADVERSARIAL ###################
#Starts a timer to track how long repeated segment takes
start_repeat = time.monotonic()

#Generator now will use data points from the discriminator to train.
ADVERSARIAL = os.environ.get("ADVERSARIAL", True)

#Resets the initial data variable so it can be used in the repeated segment.
init_data = []
#List to contain the means of errors for each run
errorlist = []
errorlist.append(errors)
#List to store the true positives found by the generator, and feed them to the discriminator
repeat_data = Positives(probs)
#Variable to count how many iterations it takes for the generator to fool the discriminator
iternum = 1

while np.mean(errorlist) not in range(48,53):
    #Repeated generator code; trains with repeat_data
    if not MODEL_OUTPUT_FILE:
      ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
      MODEL_OUTPUT_FILE = "gen_models/GRU-%s-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM, iternum)

    x_train, y_train, word_to_index, index_to_word = load_data(repeat_data, VOCABULARY_SIZE, ADVERSARIAL)

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
