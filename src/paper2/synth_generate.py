import numpy as np
import os
import random
import time
import sys
from synth_util import DirectedMultiGraph, write_info


def generate_mnl_graph(**kwargs):
    PATH = kwargs['graph_path']
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    t0 = time.time()
    num_nodes = kwargs['num_nodes']  # = 5000
    num_er_edges = kwargs['num_er_edges']  # = 25000
    num_choice_edges = kwargs['num_choice_edges']  # = 160000
    np.random.seed(kwargs['graph_seed'])

    G = DirectedMultiGraph(num_nodes)
    G.grow_erdos_renyi(num_er_edges)
    # Generate multinomial choice model with the following weights
    w0 = np.array([0.5, 2, 2, 2, 1, 4, 4, 4, 0])
    for i in range(num_choice_edges):
        # Randomly pick an actor
        actor = np.random.randint(0, G.num_nodes)
        # MNL experiment
        X = G.extract_feature_all_nodes(actor)
        U = w0.dot(X) + np.random.gumbel(size=G.num_nodes)
        U[actor] = -np.inf
        target = np.argmax(U)
        G.add_edge(actor, target)

        if i % 100 == 99:
            sys.stdout.write("\r{}/{} edges generated.".format(i + 1, num_choice_edges))
            sys.stdout.flush()

    kwargs['time_elapsed'] = time.time() - t0
    np.save(os.path.join(PATH, kwargs['er_edges_path']), G.edges_list[:num_er_edges])
    np.save(os.path.join(PATH, kwargs['choice_edges_path']), G.edges_list[num_er_edges:])
    write_info(PATH, kwargs)
    print()


def generate_mixed_mnl_graph(**kwargs):
    PATH = kwargs['graph_path']
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    t0 = time.time()
    num_nodes = kwargs['num_nodes']  # = 5000
    num_er_edges = kwargs['num_er_edges']  # = 25000
    num_choice_edges = kwargs['num_choice_edges']  # = 80000
    np.random.seed(kwargs['graph_seed'])
    G = DirectedMultiGraph(num_nodes)
    G.grow_erdos_renyi(num_er_edges)

    # weights for mixed-MNL experiment
    w0 = np.array([0.5, 1, 1, 1, 0.5, 1, 1, 1, 10000])
    w1 = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, -10000])

    for i in range(num_choice_edges):
        # Randomly pick an actor
        actor = np.random.randint(0, G.num_nodes)
        X = G.extract_feature_all_nodes(actor)
        # mixed-MNL experiment
        if np.random.rand() < 0.75:
            U = w0.dot(X) + np.random.gumbel(size=G.num_nodes)
        else:
            U = w1.dot(X) + np.random.gumbel(size=G.num_nodes)
        U[actor] = -np.inf
        target = np.argmax(U)
        G.add_edge(actor, target)
        if i % 100 == 99:
            sys.stdout.write("\r{}/{} edges generated.".format(i + 1, num_choice_edges))
            sys.stdout.flush()
    kwargs['time_elapsed'] = time.time() - t0
    np.save(os.path.join(PATH, kwargs['er_edges_path']), G.edges_list[:num_er_edges])
    np.save(os.path.join(PATH, kwargs['choice_edges_path']), G.edges_list[num_er_edges:])
    write_info(PATH, kwargs)
    print()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("[ERROR] missing graph id")
        exit()

    if sys.argv[1] == 'mnl':
        kwargs = {
            'num_nodes': 5000,
            'num_er_edges': 25000,
            'num_choice_edges': 160000,
            'graph_seed': random.randint(0, 2 ** 31 - 1),
            'graph_path': '../data/paper2_mnl',
            'er_edges_path': 'er_edges.npy',
            'choice_edges_path': 'choice_edges.npy'
        }
        generate_mnl_graph(**kwargs)

    if sys.argv[1] == 'mixed-mnl':
        kwargs = {
            'num_nodes': 5000,
            'num_er_edges': 25000,
            'num_choice_edges': 80000,
            'graph_seed': random.randint(0, 2 ** 31 - 1),
            'graph_path': '../data/paper2_mixed_mnl',
            'er_edges_path': 'er_edges.npy',
            'choice_edges_path': 'choice_edges.npy'
        }
        generate_mixed_mnl_graph(**kwargs)
