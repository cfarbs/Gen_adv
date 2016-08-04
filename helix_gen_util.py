#!/usr/bin/env python

################ DEBUG TOGGLE ####################

Debug = False

##################################################
#Discriminator packages

import sys
import pickle
from helix_network.lib.helix_neural_network import classify_with_network2
from argparse import ArgumentParser
#from multiprocessing import Process, current_process, Manager



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

################### FOR DEBUGGING ##############################
    if Debug:
        init_data = pickle.load(open("helix_network/lib/gen_helices.pkl","rb"))
        init_data = init_data[:1000]
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
        return errors, probs
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

if __name__ == "__main__":
    sys.exit(main(sys.argv))
