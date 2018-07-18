import csv
import networkx as nx
import numpy as np
import os
import pandas as pd
import random

from collections import Counter
from copy import deepcopy
from glob import glob
from multiprocessing import Pool
from scipy.optimize import minimize

from env import *

"""

  This script contains a few handy reused utility functions,
  and reused parameter settings and paths.

"""


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
