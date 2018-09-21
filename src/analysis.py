from util import *
from logit import *
import os
from multiprocessing import Pool

"""

  Driver script #3 - Fit models

  input : env.data_path/choices/
  output: ./results/

  python analyze.py > ../results/r_vs_p_synth.csv

"""


def fit_one(fn):
    """
    Fit series of models on a single data set of choices.
    Used for the analysis in section 5.1 of the paper.
    """

    # 1) fit mis-specified single log-degree model
    # TODO: remove?
    m1 = LogDegreeModel(fn, vvv=0)
    m1.fit()
    print("%s,%s,%.5f,%.3f" % (fn[:-4], "p-single", m1.u[0], m1.ll()))

    # 2) fit mis-specified mixed log-degree model
    m2 = MixedLogitModel(fn, vvv=0)
    m2.add_uniform_model()
    m2.add_log_degree_model(bounds=((1, 1),))  # clamped at alpha=1
    m2.fit(etol=0.01, n_rounds=100, return_stats=False)
    print("%s,%s,%.5f,%.3f" % (fn[:-4], "p-mixed", m2.pk[1], m2.ll()))

    # 3) fit well-specified mixed r-p model
    m3 = MixedLogitModel(fn, vvv=0)
    m3.add_uniform_model()
    m3.add_log_degree_model(bounds=((1, 1),))  # clamped
    m3.add_uniform_fof_model()
    m3.add_log_degree_fof_model(bounds=((1, 1),))  # clamped
    m3.fit(etol=0.01, n_rounds=200, return_stats=False)
    print("%s,%s,%.5f,%.3f" % (fn[:-4], "rp-mixed", m3.pk[1] + m3.pk[3], m3.ll()))


if __name__ == '__main__':
    graphs = os.listdir(data_path + "/choices")
    print("fn,model,estimate,ll")
    # run fits in parallel
    with Pool(processes=30) as pool:
        r = pool.map(fit_one, graphs)
    # TODO: write all to one file, rather than hacky pipe
