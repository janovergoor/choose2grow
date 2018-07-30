import sys
from util import *
from logit import *


"""

  Driver script #3 - Estimate

  This script takes (for each graph or set of graph) a list of choice sets,
  and fits one of different multinomial logit models. Call with:

  python driver3_estimate ${model}

  Possible models include:

    'degree_group' - separate coefficient for each degree [grouped data]
    'poly_degree_group' - k-degree polynomial for degree [grouped data]
    'log_degree_group' - single alpha coefficient for log(degree) [grouped data]
    'degree' - separate coefficient for each degree
    'uniform_degree' - uniform over all nodes
    'poly_degree' - k-degree polynomial for degree
    'log_degree' - single alpha coefficient for log(degree)
    'uniform_fof' - uniform over friends of friends
    'log_fof' - single alpha coefficient for log(# friends of friends)
    'mixed_logit' - probablistic combination of k of the aforementioned modes

  input : env.data_path/choices/
  output: env.data_path/fits/${model}

"""

if __name__ == '__main__':

    # read what kind of model to do
    if len(sys.argv) < 2:
        raise Exception("not enough command line arguments!")
    model = sys.argv[1]
    if model not in ['degree_group', 'poly_degree_group', 'log_degree_group',
                     'degree', 'uniform_degree', 'poly_degree', 'log_degree',
                     'uniform_fof', 'log_fof', 'mixed_logit']:
        raise Exception("%s not in model list.." % model)

    # read available graphs to do
    if 'group' in model:
        graphs = os.listdir(data_path + "/choices")
    else:
        graphs = os.listdir(data_path + "/choices_sampled")
    
    # make sure all the output folder is there
    mkdir('%s/fits' % data_path)
    mkdir('%s/fits/%s' % (data_path, model))

    # read what fits already done
    fits = os.listdir(data_path + "/fits/" + model)
    graphs = [x for x in graphs if x not in fits]
    # graphs  = [x for x in graphs if 'g' in x]

    # construct function to execute
    def do_one(graph):
        if 'mixed' not in model:
            if model == 'degree_group':
                m = DegreeModelGrouped(graph, vvv=1)
            elif model == 'poly_degree_group':
                m = PolyDegreeModelGrouped(graph, vvv=1)
            elif model == 'log_degree_group':
                m = LogDegreeModelGrouped(graph, vvv=1)
            elif model == 'degree':
                m = DegreeModel(graph, vvv=1)
            elif model == 'uniform_degree':
                m = UniformDegreeModel(graph, vvv=1)
            elif model == 'poly_degree':
                m = PolyDegreeModel(graph, vvv=1)
            elif model == 'log_degree':
                m = LogDegreeModel(graph, vvv=1)
            elif model == 'uniform_fof':
                m = UniformFofModel(graph, vvv=1)
            elif model == 'log_fof':
                m = LogFofModel(graph, vvv=1)
            m.fit()
            m.write_params()
        elif model == 'mixed_logit':
            m = MixedLogit(graph, vvv=1)
            # TODO - The parameters are currently clamped at 1.
            m.add_log_model(bounds=((1, 1),))
            m.add_poly_model(k=1, bounds=((1, 1),))  # uniform

    print("TODO: %d" % len(graphs))
    if sys.version_info > (3, 0):
        with Pool(processes=20) as pool:
            r = pool.map(do_one, graphs)
    else:
        from contextlib import closing
        with closing(Pool(processes=20)) as pool:
            r = pool.map(do_one, graphs)
            pool.terminate()

    # group in serial - minimize fails in parallel (at least used to)
    for gt in ['g', 'd']:
        for r in [0, 0.1, 0.25, 0.5, 0.75, 1]:
            for p in [0, 0.1, 0.25, 0.5, 0.75, 1]:
                fn = '%s-%.2f-%.2f-all.csv' % (gt, r, p)
                do_one(fn)
