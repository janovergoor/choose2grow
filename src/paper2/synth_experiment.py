from multiprocessing import Pool
import numpy as np
import random
import os
import sys
from itertools import product
from synth_util import *

data_folder = '../data/paper2_plot_data/'


def extract_feature(**kwargs):
    n = kwargs['n']
    s = kwargs['s']
    seed = kwargs['feature_seed']
    sampling = kwargs['sampling']
    edge_sampling = kwargs['edge_sampling']
    graph_path = kwargs['graph_path']
    ss = 4 if 'mixed' in graph_path else 3
    np.random.seed(seed)

    info = parse_info(graph_path)
    er_edges = np.load(os.path.join(graph_path, info['er_edges_path']), mmap_mode='r')
    choice_edges = np.load(os.path.join(graph_path, info['choice_edges_path']), mmap_mode='r')
    G = DirectedMultiGraph(info['num_nodes'])

    for actor, target in er_edges:
        G.add_edge(actor, target)
    features = []
    lnsws = []
    to_sample = set(range(n)) if edge_sampling == 'first-n' else \
                set(random.sample(range(len(choice_edges)), n))

    n1 = n2 = n3 = 1
    for i, (actor, target) in enumerate(choice_edges):
        if i in to_sample:
            candidates, lnsw = None, None
            if sampling == 'stratified':
                candidates, lnsw = G.neg_samp_by_locality(actor, target, num_neg=s, max_num_local_sample=[s // ss, s // ss])
            elif sampling == 'importance':
                s1 = int(np.floor((s - 3) * n1 / (n1 + n2 + n3)) + 1)
                s2 = int(np.floor((s - 3) * n2 / (n1 + n2 + n3)) + 1)
                candidates, lnsw = G.neg_samp_by_locality(actor, target, num_neg=s, max_num_local_sample=[s1, s2])
            else:
                candidates = [target]
                candidates.extend(randint_excluding(0, G.num_nodes, [actor, target], size=s))
                lnsw = [0.0] * len(candidates)
            feature = G.extract_feature(actor, candidates)
            if feature[5][0] + feature[6][0] > 0.5:
                n1 += 1
            elif feature[8][0] > 0.5:
                n2 += 1
            else:
                n3 += 1
            # special case for Fig 4 data
            if 'mixed' not in graph_path:
                feature = feature[:-1]
            features.append(feature.T)
            lnsws.append(lnsw)
        G.add_edge(actor, target)
    return np.array(features), np.array(lnsws)


#
# Code for Figure 3
#

def experiment(kwargs):
    X, lnsw = extract_feature(**kwargs)
    y = np.zeros(X.shape[0]).astype(int)
    m = MNLogit()
    m.data(X, y, sws=lnsw)
    m.fit(max_num_iter=500, clip=1.0, clip_norm_ord=2)
    return m.get_model_info()


def fig_3a_3b(dest=data_folder + 'fig_3ab_data.csv', nit=50):
    result = None
    cases = product(range(500, 6001, 500), [24, 96], ['uniform', 'stratified', 'importance'], range(nit))
    PATH = '../data/paper2_mnl'
    kwargs_list = [{
        'n': n,
        's': s,
        'sampling': sampling,
        'i': i,
        'graph_id': parse_info(PATH)['graph_seed'],
        'feature_seed': random.randint(0, 2 ** 31 - 1),
        'edge_sampling': 'random-uniform',
        'graph_path': PATH
    } for n, s, sampling, i in cases]
    with Pool(4) as p:
        result = p.map(experiment, kwargs_list)
    write_csv(dest, kwargs_list, result)


def fig_3c(dest=data_folder + 'fig_3c_data.csv', nit=50):
    result = None
    cases = product([3, 6, 12, 24, 48, 96, 192, 384, 768], ['uniform', 'stratified', 'importance'], range(nit))
    PATH = '../data/paper2_mnl'
    kwargs_list = [{
        'n': 10000,
        's': s,
        'sampling': sampling,
        'i': i,
        'graph_id': parse_info(PATH)['graph_seed'],
        'feature_seed': random.randint(0, 2 ** 31 - 1),
        'edge_sampling': 'random-uniform',
        'graph_path': PATH
    } for s, sampling, i in cases]
    with Pool(4) as p:
        result = p.map(experiment, kwargs_list)
    write_csv(dest, kwargs_list, result)


def fig_3d(dest=data_folder + 'fig_3d_data.csv', nit=50):
    result = None
    cases = product([3, 6, 12, 24, 48, 96, 192, 384, 768], ['uniform', 'stratified', 'importance'], range(nit))
    PATH = '../data/paper2_mnl'
    kwargs_list = [{
        'n': 480000 // s,
        's': s,
        'sampling': sampling,
        'i': i,
        'graph_id': parse_info(PATH)['graph_seed'],
        'feature_seed': random.randint(0, 2 ** 31 - 1),
        'edge_sampling': 'random-uniform',
        'graph_path': PATH
    } for s, sampling, i in cases]
    with Pool(4) as p:
        result = p.map(experiment, kwargs_list)
    write_csv(dest, kwargs_list, result)


#
# Code for Figure 4
#

def experiment_1cl(kwargs):
    X, lnsw = extract_feature(**kwargs)
    y = np.zeros(X.shape[0]).astype(int)
    m = MNLogit()
    m.data(X[..., :-1], y, sws=lnsw)
    m.fit(max_num_iter=500, clip=1.0, clip_norm_ord=2)
    return m.get_model_info()


def experiment_2cl(kwargs):
    X, lnsw = extract_feature(**kwargs)
    y = np.zeros(X.shape[0]).astype(int)
    ffof = X[:, 0, 8]
    ind_1 = ffof > 0.5  # local (ffof)
    ind_2 = ffof < 0.5  # non-local
    m1 = MNLogit()
    sws_filter_1 = X[ind_1, :, 8] < 0.5
    m1.data(X[ind_1, :, :-1], y[ind_1], sws=lnsw[ind_1] - 10000 * sws_filter_1)
    m1.fit(max_num_iter=500, clip=1.0, clip_norm_ord=2)
    info1 = m1.get_model_info()
    m2 = MNLogit()
    sws_filter_2 = X[ind_2, :, 8] > 0.5
    m2.data(X[ind_2][..., [0, 4]], y[ind_2], sws=lnsw[ind_2] - 10000 * sws_filter_2)
    m2.fit(max_num_iter=500, clip=1.0, clip_norm_ord=2)
    info2 = m2.get_model_info()
    merged_info = {k + "_1": info1[k] for k in info1}
    merged_info.update({k + "_2": info2[k] for k in info1})
    return merged_info


def fig_4a_4b(dest=data_folder + 'fig_4ab_data.csv', nit=20):
    result = None
    cases = product([16, 32, 64, 128, 256, 512, 1024], ['uniform', 'stratified'], range(nit))
    PATH = '../data/paper2_mixed_mnl'
    kwargs_list = [{
        'n': 80000,
        's': s,
        'sampling': sampling,
        'i': i,
        'graph_id': parse_info(PATH)['graph_seed'],
        'feature_seed': random.randint(0, 2 ** 31 - 1),
        'edge_sampling': 'first-n',
        'graph_path': PATH
    } for s, sampling, i in cases]
    with Pool(48) as p:
        result = p.map(experiment_1cl, kwargs_list)
    write_csv(dest, kwargs_list, result)


def fig_4c_4d(dest=data_folder + 'fig_4cd_data.csv', nit=20):
    result = None
    cases = product([16, 32, 64, 128, 256, 512, 1024], ['uniform', 'stratified'], range(nit))
    PATH = '../data/paper2_mixed_mnl'
    kwargs_list = [{
        'n': 80000,
        's': s,
        'sampling': sampling,
        'i': i,
        'graph_id': parse_info(PATH)['graph_seed'],
        'feature_seed': random.randint(0, 2 ** 31 - 1),
        'edge_sampling': 'first-n',
        'graph_path': PATH
    } for s, sampling, i in cases]
    with Pool(48) as p:
        result = p.map(experiment_2cl, kwargs_list)
    with open(dest, 'w') as f:
        header = ['num_nodes', 'graph_id', 'sampling', 'data_size', 's', 'run_id']
        header.extend(['w_local_{}'.format(i) for i in range(8)])
        header.extend(['se_local_{}'.format(i) for i in range(8)])
        header.extend(['w_nonlocal_{}'.format(i) for i in range(2)])
        header.extend(['se_nonlocal_{}'.format(i) for i in range(2)])
        f.write(','.join(header) + '\n')
        for info, kwargs in zip(result, kwargs_list):
            row = [5000, kwargs['graph_id'], kwargs['sampling'], kwargs['n'], kwargs['s'], kwargs['i']]
            row.extend(info['weights_1'])
            row.extend(info['se_1'])
            row.extend(info['weights_2'])
            row.extend(info['se_2'])
            f.write(','.join(map(str, row)) + '\n')


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("[ERROR] missing graph id")
        exit()

    if sys.argv[1] == 'fig3':
        fig_3a_3b(nit=100)
        fig_3c(nit=100)
        fig_3d(nit=100)

    if sys.argv[1] == 'fig4':
        fig_4a_4b(nit=100)
        fig_4c_4d(nit=100)
