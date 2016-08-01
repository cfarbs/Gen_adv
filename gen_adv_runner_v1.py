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
#from multiprocessing import Process, current_process, Manager

#################### INITIAL GENERATOR CODE ###############################

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "24"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))
#Need to figure out how I'm going to start with one infile and change to another
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "GRU/rawhelices.pkl")

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
    parser.add_argument('adversarial', '-ad', action='store', dest='out',
                        required=True, type=str, default=True,
                        help="determines whether run will be adversarial")

    args = parser.parse_args()
    #print (type(args))
    return args

#def run_nn2(work_queue, done_queue):
#    try:
#        for f in iter(work_queue.get, 'STOP'):
#            n = classify_with_network2(**f)
#    except Exception:
#        done_queue.put("%s failed" % current_process().name)

def main(args):
    args = parse_args()

    config = pickle.load(open(args.config, 'rb'))

    try:
            extra_args = config['extra_args']
            batch_size = extra_args['batch_size']
    except KeyError:
            extra_args = None
            batch_size = args.batch_size

    assert(batch_size is not None), "You need to specify batch_size with a flag or have it in the config file"

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
#    workers = args.jobs
#    work_queue = Manager().Queue()
#    done_queue = Manager().Queue()
#    jobs = []

    for experiment in range(len(config['helixdict'])):
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
        errors, probs = classify_with_network2(**nn_args)  # activate for debugging
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

    print (sys.stderr, "\n\tFinished Neural Net")
    print (sys.stdout, "\n\tFinished Neural Net")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
