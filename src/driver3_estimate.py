import sys
from util import *
from logit import *


"""

  Driver script #3 - Estimate

  This script takes (for each graph or set of graph) a list of choice sets,
  and fits one of different multinomial logit models. Call with:

  python driver3_estimate ${model}

  Possible models include:

    'logit_degree_group' - separate coefficient for each degree [grouped data]
    'logit_poly_group' - k-degree polynomial for degree [grouped data]
    'logit_log_group'- single alpha coefficient for log(degree) [grouped data]
    'logit_degree' - separate coefficient for each degree [individual data]
    'logit_poly' - k-degree polynomial for degree [individual data]
    'logit_log'- single alpha coefficient for log(degree) [individual data]
    'mixed_logit' - probablistic combination of k of the aforementioned modes

  input : env.data_path/choices/
  output: env.data_path/fits/${model}

"""

if __name__ == '__main__':

    # read what kind of model to do
    if len(sys.argv) < 2:
        raise Exception("not enough command line arguments!")
    model = sys.argv[1]
    if model not in ['logit_degree_group', 'logit_poly_group', 'logit_log_group',
                     'logit_degree', 'logit_poly', 'logit_log', 'mixed_logit']:
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
            if model == 'logit_degree_group':
                m = DegreeLogitModelGrouped(graph, vvv=1)
            elif model == 'logit_poly_group':
                m = PolyLogitModelGrouped(graph, vvv=1)
            elif model == 'logit_log_group':
                m = LogLogitModelGrouped(graph, vvv=1)
            elif model == 'logit_degree':
                m = DegreeLogitModel(graph, vvv=1)
            elif model == 'logit_poly':
                m = PolyLogitModel(graph, vvv=1)
            elif model == 'logit_log':
                m = LogLogitModel(graph, vvv=1)
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
