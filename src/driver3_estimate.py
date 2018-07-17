from logit_grouped import *
from util import *

"""

  Driver script #3 - Estimate

  This script takes (for each graph or set of graph) a list of choice sets,
  and fits one of different multinomial logit models. Possible modes include:

  * DegreeLogit - separate coefficient for each degree
  * LogLogit - single alpha coefficient for log(degree)
  * PolyLogit - k-dimensional polynomial for degree
  * MixedLogit - probablistic combination of k of the aforementioned modes

  input : env.data_path/choices/
  output: env.data_path/fits_param/
          env.data_path/fits_util/

"""

# make sure all the output folders are there
for folder in ['fits_utility', 'fits_param']:
    mkdir('%s/%s' % (data_path, folder))


def do_one_degree(graph):
    m = DegreeLogitModel(graph, vvv=2)
    m.fit()
    m.write_degree_utilities()
    m.write_params()


def do_one_log(graph):
    m = LogLogitModel(graph, vvv=2)
    m.fit()
    m.write_degree_utilities()
    m.write_params()


def do_one_poly(graph):
    m = PolyLogitModel(graph, vvv=2)
    m.fit()
    m.write_degree_utilities()
    m.write_params()


def do_one_mixed_logit(graph):
    # TODO - The parameters are currently clamped at 1.
    m = MixedLogitModel(graph, vvv=1)
    m.add_log_model(bounds=((1, 1),))
    m.add_poly_model(k=1, bounds=((1, 1),))  # uniform
    m.fit(n_rounds=250, etol=10)
    m.write_degree_utilities()
    m.write_params()


if __name__ == '__main__':

    # individual in parallel
    graphs = os.listdir(data_path + "/choices")  # todo
    fits = os.listdir(data_path + "/fits_param/logit_degree")  # already done
    graphs = [x for x in graphs if x not in fits]
    # graphs  = [x for x in graphs if 'g' in x]
    print("TODO: %d" % len(graphs))
    with Pool(processes=20) as pool:
        r = pool.map(do_one_degree, graphs)
        # r = pool.map(do_one_mixed_logit, graphs)

    # group in serial - minimize fails in parallel (at least used to)
    for gt in ['g', 'd']:
        for r in [0, 0.1, 0.25, 0.5, 0.75, 1]:
            for p in [0, 0.1, 0.25, 0.5, 0.75, 1]:
                fn = '%s-%.2f-%.2f-all.csv' % (gt, r, p)
                do_one_degree(fn)
                # do_one_mixed_logit(fn)
