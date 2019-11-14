'''
'''
from __future__ import print_function, division

import pickle
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tasks import create_train_test_graphs, test_edge_functions, plot_parameter_sensitivity, grid_search

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task', type=str,
                        help="Task to run, one of 'gridsearch', 'edgeencoding', and 'sensitivity'")

    parser.add_argument('--input', nargs='?', default='./input/graph.txt',
                        help='Input graph path')

    parser.add_argument('--regen', dest='regen', action='store_true',
                        help='Regenerate random positive/negative links')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers. Default is 1.')

    parser.add_argument('--num_experiments', type=int, default=5,
                        help='Number of experiments to average. Default is 5.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.task is None:
        print("Specify task to run: edgeembedding, sensitivity, gridsearch")
        exit()

    if args.task.startswith("grid"):
        grid_search(args)

    elif args.task.startswith("edge"):
        test_edge_functions(args)

    elif args.task.startswith("sens"):
        plot_parameter_sensitivity(args)