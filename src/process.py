import networkx as nx
from collections import Counter
from multiprocessing import Pool
from network_stats import *
from util import *

"""

  Driver script #2 - Convert edge data to choice data

  This processes all graphs in `env.graphs_path`. For each graph,
  the following data sets are produced:

  * /edges - relevant network context features for every edge
  * /choices - for every edge, sampled choice set data (degree, # fof)

  input : env.graphs_path/
  output: env.data_path/edges/
          env.data_path/choices/

"""

# make sure all the output folders are there
for folder in ['edges', 'choices']:
    mkdir('%s/%s' % (data_path, folder))


def fof_paths(G, i):
    """
    Count the number of length-2 paths to each node j starting from i,
    removing anyone in i's neighborhood.
    """
    fofs = {}
    neighbors = list(nx.neighbors(G, i))
    for k in neighbors:
        for j in nx.neighbors(G, k):
            # skip neighbors of i, or i itself
            if j in neighbors or j == i:
                continue
            if j not in fofs:
                fofs[j] = 0
            fofs[j] += 1
    return fofs


def process_all_edges(graph, n_alt=10, vvv=0):
    """
    Read in a graph from an edge list, compute for every edge the network
    context when that edge was formed, and write out the results.

    'graph' is a filename
    """
    # read the edge list from file
    el = read_edge_list(graph, vvv)
    # create initial graph
    G = nx.Graph()
    for (t, i, j) in el:
        if t == 0:
            G.add_edge(i, j)
    if vvv > 1:
        print("[%s] initial graph has %d nodes and %d edges" %
              (graph, len(G.nodes()), len(G.edges())))
    # look at every edge individually
    T = []
    for (t, i, j) in el:
        if t == 0:
            continue
        G.add_nodes_from([i, j])  # add nodes
        # create degree distribution dictionary
        degs = Counter(dict(G.degree()).values())
        # create # fof paths dictionary
        paths = fof_paths(G, i)
        # retrieve fof_deg for j
        fof_deg = paths[j] if j in paths else 0
        # grab all friends
        friends = set(nx.ego_graph(G, i, 1).nodes())
        # construct set of nodes that are eligible to be selected
        eligible_js = set(G.nodes()) - friends
        # construct set of nodes that are eligible to be selected AND fof
        eligible_fofs = set(nx.ego_graph(G, i, 2).nodes()) - friends
        # sample $n_alt negative examples and add positive example
        choices = list(eligible_js - set([j]))
        # NOTE: currently sampling with replacement
        mln_sample = np.random.choice(choices, n_alt, replace=True)
        mln_sample = list(mln_sample) + [j]
        # add all fields for the current edge
        T.append({
            't': t,
            'i': i,
            'j': j,
            'n': G.number_of_nodes(),
            # number of eligible nodes to connect to
            'n_elig': len(eligible_js),
            # number of eligible nodes that are friends of friends
            'n_elig_fof': len(eligible_fofs),
            # degree of the connected to node (j)
            'deg_i': G.degree(i),
            # number of nodes with degree 'deg'
            'n_deg_i': float(degs[G.degree(i)]),
            # degree of the connected to node (j)
            'deg_j': G.degree(j),
            # number of nodes with degree 'deg'
            'n_deg_j': float(Counter(dict(G.degree()).values())[G.degree(j)]),
            # number of length-2 paths between i and j
            'fof_deg': fof_deg,
            # number of nodes with same amount of fof_paths
            'n_fof_deg': len([x[0] for x in paths.items() if x[1] == fof_deg]),
            # total length-2 paths from i
            'tot_fof': sum(paths.values()),
            # total degree distribution
            'degree_distribution': Counter(list(dict(G.degree()).values())),
            # negative sampled data for MLN estimation
            'mln_data': [(1 if k == j else 0,  # chosen?
                          G.degree(k),  # degree
                          paths[k] if k in paths else 0  # number of FoF paths
                          ) for k in mln_sample]
        })
        # actually add the edge
        G.add_edge(i, j)
        if vvv > 1:
            if len(G.edges()) % 1000 == 0:
                print("[%s] done %d/%d edges" %
                      (graph, len(G.edges()), len(el)))
    # print final statement
    if vvv > 0:
        print("[%s] final graph has %d nodes and %d edges" %
              (graph, len(G.nodes()), len(G.edges())))

    # write individual edges
    cols = ['t', 'i', 'j', 'n', 'n_elig', 'n_elig_fof', 'deg_i', 'n_deg_i',
            'deg_j', 'n_deg_j', 'fof_deg', 'n_fof_deg', 'tot_fof']
    fn = '%s/%s/%s' % (data_path, 'edges', graph)
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for e in T:
            writer.writerow([e[x] for x in cols])
    if vvv > 1:
        print("[%s] did edges" % graph)

    # write negatively sampled (degree, fof) choice sets
    fn = '%s/%s/%s' % (data_path, 'choices', graph)
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['choice_id', 'y', 'deg', 'fof'])
        for e in range(len(T)):
            for (y, deg, fof) in T[e]['mln_data']:
                writer.writerow([e, y, deg, fof])
    if vvv > 0:
        print("[%s] did sampled choices" % graph)


if __name__ == '__main__':
    graphs = os.listdir(graphs_path)  # todo
    choices = os.listdir(data_path + "/choices")  # already done
    graphs = [x for x in graphs if x not in choices]
    graphs = [x for x in graphs if 'all' not in x]  # remove school networks
    print("TODO: %d" % len(graphs))
    with Pool(processes=20) as pool:
        r = pool.map(process_all_edges, graphs)
