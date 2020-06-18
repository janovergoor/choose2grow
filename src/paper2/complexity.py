import csv
import time
import pandas as pd
import numpy as np
import logit
from multiprocessing import Pool
pd.set_option('mode.chained_assignment', None)  # danger


"""

  Time complexity analysis

  This script generates, for each combination of (n, s), a synthetic dataset
  of n choices with s alternatives each, with p=5 features generated uniformly
  at random. Then it fits a conditional logit model and records the runtime.

"""


def do_one(case):
    (n, s) = case
    p = 5  # number of features
    features = ['x' + str(i) for i in range(p)]
    columns = ['choice_id', 'j', 'y'] + features
    data = []
    for choice_id in np.arange(n):
        for j in np.arange(s):
            data.append([choice_id, j, 1 if j == 0 else 0] + list(np.random.randint(0, 9, p) / 1000))
    D = pd.DataFrame(data, columns=columns)
    ids = ','.join([str(x) for x in case])
    m = logit.FeatureModel(ids, D=D, vvv=0, features=features)
    t = time.time()
    m.fit()
    t = time.time() - t
    return list(case) + [p, t]


def run_complexity():
    # define cases to run
    cases = []
    for n in [10, 30, 100, 300, 1000, 3000]:
        for s in [1, 3, 10, 30, 100, 300, 1000]:
            if n * s > 100000000:
                continue
            for i in range(50):
                cases.append((n, s))
    # run processes in parallel
    with Pool(processes=30) as pool:
        res = pool.map(do_one, cases)
    # write results
    with open("../data/paper2_plot_data/fig_2_data.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 's', 'p', 'time'])
        for row in res:
            writer.writerow(row)
