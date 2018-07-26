import csv
import networkx as nx
import numpy as np
import os
import pandas as pd
import scipy as sp
import random
import sys

from collections import Counter
from copy import deepcopy
from glob import glob
from multiprocessing import Pool

from env import *

"""

  This script contains a few handy reusable utility functions,
  and parameter settings and paths.

"""

# amount to smooth log(0) with
log_smooth = 1e-8


# Helper functions

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


def write_edge_list(T, fn):
    """
    Write a time-stamped edge list to file.
    """
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'from', 'to'])
        for (t, i, j) in T:
            writer.writerow([t, i, j])


def check_grad_rel(func, grad, x0, *args):
    """
    Does a relative check of the gradient.
    Uses scipy.optimize.approx_fprime
    """
    step = 1.49e-08
    target = sp.optimize.approx_fprime(x0, func, step, *args)
    actual = grad(x0, *args)
    delta = target - actual
    # make sure target is not 0
    delta[target > 0] /= target[target > 0]
    return delta

