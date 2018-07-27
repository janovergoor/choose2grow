from network_generation import *
from util import *

"""

  Driver script #1 - Generate

  This script generates (for each combination of value for r and p),
  a number of graphs, and writes them out as ordered edge lists.
  The structure of the filename is:

      "%s-%.2f-%.2f-%.02d.csv" % ({d,g}, r, p, id)

  Each id is generated from the *same* seed graph.

  output: env.graphs_path/

"""

# make sure the output folder exists
mkdir(graphs_path)

# target number of nodes for each value of p
ns = {0.00: 22500,
      0.10: 20000,
      0.25: 17500,
      0.50: 15000,
      0.75: 12500,
      1.00: 10000}


def do_one_dense_cycle(id):
    """
    Generate a small random graph, and for each value of r and p,
    generate a "mixed model" with a densifying process.
    The same graph is reused for every value of (r,p)!
    """
    Gt = nx.erdos_renyi_graph(1000, 0.005)
    for p in [0, 0.1, 0.25, 0.5, 0.75, 1]:
        for r in [0, 0.1, 0.25, 0.5, 0.75, 1]:
            fn = '%s/d-%.2f-%.2f-%.02d.csv' % (graphs_path, r, p, id)
            print(fn)
            (G, el) = generate_mixed_model(fn, graph=Gt, n_max=ns[p], r=r, p=p, grow=False)
            write_edge_list(el, fn)


def do_one_grow_cycle(id):
    """
    For each value of r and p, generate a "mixed model" with a growing process.
    """
    for p in [0, 0.1, 0.25, 0.5, 0.75, 1]:
        for r in [0, 0.1, 0.25, 0.5, 0.75, 1]:
            fn = '%s/g-%.2f-%.2f-%.02d.csv' % (graphs_path, r, p, id)
            print(fn)
            (G, el) = generate_mixed_model(fn, n_max=ns[p], r=r, p=p, grow=True, m=4)
            write_edge_list(el, fn)


if __name__ == '__main__':
    # Run cycles in parallel
    with Pool(processes=10) as pool:
        r = pool.map(do_one_dense_cycle, range(100))
    with Pool(processes=10) as pool:
        r = pool.map(do_one_grow_cycle, range(100))
