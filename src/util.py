import csv
import numpy as np
import os
import pandas as pd
import random
from glob import glob
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


def manual_ll(D, alpha=1, p=0.5):
    """
    Manually compute the log likelihood for a model where edges are formed
    with by preferential attachment with alpha with probability p and
    uniformly at random with probability 1-p.
    """
    # preferential attachment
    # transform degree to score
    D['score'] = np.exp(alpha * np.log(D.deg + log_smooth))
    # compute total utility per case
    score_tot = D.groupby('choice_id')['score'].aggregate(np.sum)
    # compute probabilities of choices
    scores_pa = np.array(D.loc[D.y == 1, 'score']) / np.array(score_tot)
    # uniform
    scores_uniform = np.array(1.0 / D.groupby('choice_id')['y'].aggregate(len))
    # combine stores
    scores = p * scores_pa + (1 - p) * scores_uniform
    # add tiny smoothing for deg=0 choices
    scores += log_smooth
    # return sum
    return -1 * sum(np.log(scores))


#
#  Data reading functions
#


def read_edge_list(graph, vvv=0):
    """
    Read a time-steamped edge list from a csv file.
    The expected input format is: ['ts', 'from', 'to']
    For FB data, the format is: ['i', 'j', 'ts', 'direction']
    """
    # read in the edge list
    el = []
    with open('%s/%s' % (graphs_path, graph), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # skip header
        next(reader, None)
        for row in reader:
            # FB networks are unordered but with direction
            if len(row) == 4:
                # switch if necessary
                if row[3] == '2':
                    row = [row[2], row[1], row[0]]
                else:
                    row = [row[2], row[0], row[1]]
            el.append([int(x) for x in row])
    if vvv > 0:
        print("[%s] read %d edges" % (graph, len(el)))
    return el


def read_data(fn, max_deg=50, vvv=0):
    """
    Read individual data for either a single graph's choice sets,
    or all graphs with the specified parameters.
    """
    if 'all' in fn:
        # read all
        Ds = []
        # get all files that match
        fn_tmp = '-'.join(fn.split('-')[:-1])
        pattern = "%s/choices/%s*.csv" % (data_path, fn_tmp)
        fns = [os.path.basename(x) for x in glob(pattern)]
        for x in fns:
            path = '%s/%s/%s' % (data_path, 'choices', x)
            D = read_data_single(path, max_deg)
            # update choice ids so they dont overlap
            fid = x.split('.csv')[0].split('-')[-1]
            ids = [('%09d' + fid) % x for x in D.choice_id]
            D.choice_id = ids
            Ds.append(D)
        # append the results
        D = np.hstack(Ds)
    else:
        # read one
        path = '%s/%s/%s' % (data_path, 'choices', fn)
        D = read_data_single(path, max_deg)
    # cut off at max observed degree
    if vvv:
        print("[%s] read (%d x %d)" % (fn, D.shape[0], D.shape[1]))
    return D


def read_data_single(path, max_deg=50):
    """
    Read individual data for a single graph.
    Degrees are cut-off at max_deg.
    """
    # read the choices
    D = pd.read_csv(path)
    if max_deg is not None:
        # remove too high degree choices
        D = D[D.deg <= max_deg]
    # remove cases without any choice (choice was higher than max_deg)
    D = D[D.groupby('choice_id')['y'].transform(np.sum) == 1]
    # read the choices
    return D
