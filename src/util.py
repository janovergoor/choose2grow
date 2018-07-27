import csv
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import sys

from glob import glob
from multiprocessing import Pool
from scipy.optimize import approx_fprime

from env import *
"""

  This script contains a few handy reusable utility functions,
  and parameter settings and paths.

"""

# amount to smooth log(0) with
log_smooth = 1e-8


#
# Helper functions
#

def mkdir(path):
    """
    Make directory, if it doesn't exist already.
    """
    if not os.path.exists(path):
        os.mkdir(path)


def random_sample(candidates):
    """
    Short-hand to draw a single random element from a list.
    """
    candidates = list(candidates)
    if len(candidates) == 0:
        return None
    return random.sample(candidates, 1)[0]


def poly_utilities(n, theta):
    """
    Generate utilities of the form: u[i] = sum_k (i^k * theta[i])
    """
    u = np.array([0.0] * n)
    for i in range(len(theta)):
        u += np.array(range(n))**i * theta[i]
    return u


def check_grad_rel(func, grad, x0, *args):
    """
    Does a relative check of the gradient.
    Uses scipy.optimize.approx_fprime
    """
    step = 1.49e-08
    target = approx_fprime(x0, func, step, *args)
    actual = grad(x0, *args)
    delta = target - actual
    # make sure target is not 0
    delta[target > 0] /= target[target > 0]
    return delta


#
#  Data reading functions
#

def write_edge_list(T, fn):
    """
    Write a time-stamped edge list to file.
    Output format is: ['t', 'from', 'to']
    """
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'from', 'to'])
        for (t, i, j) in T:
            writer.writerow([t, i, j])


def read_edge_list(graph):
    """
    Read a time-steamped edge list to file.
    Regular input format is: ['ts', 'from', 'to']
    For FB data, format is : ['i', 'j', 'ts', 'direction']
    """
    # read in the edge list
    el = []
    sep = ',' if synth else '\t'
    with open('%s/%s' % (graphs_path, graph), 'r') as f:
        reader = csv.reader(f, delimiter=sep)
        if synth:
            next(reader, None)  # skip header
        for row in reader:
            # FB networks are unordered but with direction
            if len(row) == 4:
                # switch if necessary
                if row[3] == '2':
                    row = [row[2], row[1], row[0]]
                else:
                    row = [row[2], row[0], row[1]]
            el.append([int(x) for x in row])

    print("[%s] read %d edges" % (graph, len(el)))
    return el


def read_grouped_data(fn, max_deg=50, vvv=False):
    """
    Read grouped data for either a single graph's choice sets,
    or all graphs with the specified parameters.
    Degrees are cut-off at max_deg.
    """
    if 'all' in fn:
        # read all
        Ns = []
        Cs = []
        # get all files that match
        fn_tmp = '-'.join(fn.split('-')[:-1])
        pattern = "%s/choices/%s*.csv" % (data_path, fn_tmp)
        fns = [os.path.basename(x) for x in glob(pattern)]
        for x in fns:
            (N, C) = read_grouped_data_single(x, max_deg)
            Ns.append(N)
            Cs.append(C)
        # append the results
        N = np.vstack(Ns)
        C = np.hstack(Cs)
    else:
        # read one
        (N, C) = read_grouped_data_single(fn, max_deg)
    # cut off at max observed degree
    md = np.max(np.arange(max_deg + 1)[np.sum(N, axis=0) > 0])
    N = N[:, :(md + 1)]
    if vvv:
        print("[%s] read (%d x %d)" % (fn, N.shape[0], N.shape[1]))
    return (N, C)


def read_grouped_data_single(fn, max_deg=50):
    """
    Read data (options and choices) for a single graph.
    If the max observed degree for a graph is less than
    max_deg, fill it in with zeros.
    """
    path = '%s/%s/%s' % (data_path, 'choices', fn)
    # read the choices
    dg = pd.read_csv(path)
    # remove too high degree choices
    dg = dg[dg.deg <= max_deg]
    # remove cases without any choice (choice was higher than max_deg)
    dg = dg[dg.groupby('choice_id')['c'].transform(np.sum) == 1]
    # convert counts to matrix
    ids = sorted(list(set(dg['choice_id'])))  # unique choices
    did = dict([(ids[x], x) for x in range(len(ids))])  # dictionary
    xs = [did[x] for x in dg.choice_id]  # converted indices
    # construct the matrix
    N = np.zeros((len(ids), max_deg + 1))
    N[xs, dg.deg] = dg.n
    # convert choices to vector
    C = np.array(dg[dg.c == 1].deg)
    return (N, C)


def read_individual_data(fn, max_deg=50, vvv=False):
    """
    Read individual data for either a single graph's choice sets,
    or all graphs with the specified parameters.
    """
    if 'all' in fn:
        # read all
        Ds = []
        # get all files that match
        fn_tmp = '-'.join(fn.split('-')[:-1])
        pattern = "%s/choices_sampled/%s*.csv" % (data_path, fn_tmp)
        fns = [os.path.basename(x) for x in glob(pattern)]
        for x in fns:
            D = read_individual_data_single(x, max_deg)
            # update choice ids so they dont overlap
            fid = x.split('.csv')[0].split('-')[-1]
            ids = [('%09d' + fid) % x for x in D.choice_id]
            D.choice_id = ids
            Ds.append(D)
        # append the results
        D = np.hstack(Ds)
    else:
        # read one
        D = read_individual_data_single(fn, max_deg)
    # cut off at max observed degree
    if vvv:
        print("[%s] read (%d x %d)" % (fn, D.shape[0], D.shape[1]))
    return D


def read_individual_data_single(fn, max_deg=50):
    """
    Read individual data for a single graph.
    Degrees are cut-off at max_deg.
    """
    path = '%s/%s/%s' % (data_path, 'choices_sampled', fn)
    # read the choices
    D = pd.read_csv(path)
    if max_deg is not None:
        # remove too high degree choices
        D = D[D.deg <= max_deg]
        # remove cases without any choice (choice was higher than max_deg)
        D = D[D.groupby('choice_id')['y'].transform(np.sum) == 1]
    # read the choices
    return D


