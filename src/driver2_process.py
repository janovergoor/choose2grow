from network_stats import *
from util import *

"""

  Driver script #2 - Process

  This processes all graphs in `env.graphs_path`. For each graph,
  it reads it in, and produces the following processed data sets:

  * /stats - general network statistics of the final graph
  * /edges - relevant network context features for every edge
  * /degree - the degree for every node
  * /choices - for every edge, complete choice set information (degree only)
  * /choices_sampled - for every edge, sampled choice set information (degree, # fof)

  input : env.graphs_path/
  output: env.data_path/stats/
          env.data_path/edges/
          env.data_path/degrees/
          env.data_path/choices/
          env.data_path/choices_sampled/

"""

# make sure all the output folders are there
for folder in ['stats', 'degrees', 'edges', 'choices', 'choices_sampled']:
    mkdir('%s/%s' % (data_path, folder))


if __name__ == '__main__':
    graphs = os.listdir(graphs_path)  # todo
    choices = os.listdir(data_path + "/choices_sampled")  # already done
    graphs = [x for x in graphs if x not in choices]
    graphs = [x for x in graphs if 'all' not in x]  # remove school networks
    print("TODO: %d" % len(graphs))
    with Pool(processes=10) as pool:
        r = pool.map(process_all_edges, graphs)
