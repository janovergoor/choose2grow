import networkx as nx
from multiprocessing import Pool
from util import *

"""

  Driver script #1 - Generate synthetic graphs

  This script generates (for each combination (r,p),
  a number of graphs, and writes them out as ordered edge lists.
  The structure of the filename is:

      "%s-%.2f-%.2f-%.02d.csv" % ({d,g}, r, p, id)

  Each id is generated from the *same* seed graph.

  output: data_path/synth_graphs

"""

# make sure the output folder exists
mkdir(data_path + '/synth_graphs')

#@profile
def make_rp_graph(id, G_in=None, n_max=10000, r=0.5, p=0.5, grow=True, m=1):
    """
    Generate a graph with n_max edges according to the r-p model, which
    forms edges according to both preferential attachment and triadic closure.

    If grow=True, the graph will be grown (using G_in as seed) as a
    starting graph. For every new node i, m new edges will be generated.
    If grow=False, only new edges will created between existing nodes,
    using the seed graph G_in. Node i of the edge is drawn uniformly at random.
    In either case, the process stops when n_max edges have been formed.

    For every new edge, random sample from two uniform random variables (P, R)
    and create a new edge between i and j, with j selected as follows:
    P>p & R<r - draw j uniformly at random
    P>p & R>r - draw j uniformly at random from the set of i's FoFs
    P<p & R<r - draw j based on j's degree
    P<p & R>r - draw k based on j's degree from the set of i's FoFs

    The function returns an ordered edge list.
    """
    # if no input graph is given, start with a default graph
    if G_in is None:
        if grow:
            # small complete graph for grow
            G = nx.complete_graph(5, create_using=nx.DiGraph())
        else:
            # sparse ER graph for dense
            G = nx.erdos_renyi_graph(1000, 0.005, directed=True)
    else:
        # else, copy the input graph
        G = G_in.copy()
    # store edges
    T = [(0, e[0], e[1]) for e in G.edges()]
    # set m to 1 if grow=False, as each edges is created individually
    if not grow:
        m = 1
    # count total edges created
    m_total = 1
    while len(G.edges()) < n_max:
        # select source node of the new edge(s)
        if grow:
            # add a new node
            i = max(G.nodes()) + 1
            G.add_node(i)
        else:
            # sample a random node
            i = random_sample(G.nodes())
        # count how many edges for this node i
        m_node = 0
        # pre-fill node-sets
        # gather all distinct friends
        friends = set(nx.ego_graph(G, i, 1).nodes())
        # all nodes except for friends and self are eligible
        eligible = set(G.nodes()) - friends - set([i])
        # subtract friends from FoFs to get eligible FoFs
        ego2 = set(nx.ego_graph(G, i, 2).nodes())
        eligible_fofs = ego2 - friends
        # first round, not selected yet
        j = None
        # create each edge separately
        while m_node < m:
            # intermediate print-out
            if len(G.edges()) % 5000 == 0:
                print("Done %d" % len(G.edges()))
            # add j to node sets if it is defined
            if j is not None:
                friends = friends.union(set([j]))
                eligible = eligible - set([j])
                ego2 = ego2.union(G.successors(j))
                eligible_fofs = ego2 - friends
            # don't do anything if there are no eligible nodes
            if len(eligible) == 0:
                m_node = m
                continue
            # sample from uniform random variables
            (P, R) = (random.random(), random.random())
            if R < r or len(eligible_fofs) == 0:
                # sample from full set of eligible nodes
                choice_set = eligible
            else:
                # sample from FoFs only
                choice_set = eligible_fofs
            if P > p:
                # sample uniformly
                j = random_sample(choice_set)
            else:
                if len(choice_set) == 1:
                    j = list(choice_set)[0]
                else:
                    # sample according to degree
                    ds = dict(G.in_degree(choice_set))
                    # add 1 to give new nodes a chance
                    vals = [x + 1 for x in ds.values()]
                    sum_vals = sum(vals) * 1.0
                    ps = [x / sum_vals for x in vals]
                    # could switch to np.random.multinomial
                    j = np.random.choice(list(ds.keys()), size=1, p=ps)[0]
            # check if edge didn't happen, already exists, or is a self-edge
            if j is None or G.has_edge(i, j) or i == j:
                if grow:
                    continue
                else:
                    break
            # add the edge to the edge list and graph
            T.append((m_total, i, j))
            G.add_edge(i, j)
            m_total += 1
            m_node += 1
    # return the final graph
    return (G, T)


def write_edge_list(T, fn):
    """
    Write a time-stamped edge list to a csv file.
    Output format is: ['t', 'from', 'to']
    """
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'from', 'to'])
        for (t, i, j) in T:
            writer.writerow([t, i, j])


def do_one_dense_cycle(id):
    """
    Generate a small random graph, and for each value of r and p,
    generate an r-p graph with a densifying process.
    The same graph is reused for every value of (r,p).
    """
    Gt = nx.erdos_renyi_graph(1000, 0.005, directed=True)
    for p in [0, 0.25, 0.5, 0.75, 1]:
        for r in [0, 0.25, 0.5, 0.75, 1]:
            fn = '%s/synth_graphs/d-%.2f-%.2f-%.02d.csv' % (data_path, r, p, id)
            print(fn)
            (G, el) = make_rp_graph(fn, G_in=Gt, grow=False, r=r, p=p)
            write_edge_list(el, fn)


def do_one_grow_cycle(id):
    """
    For each value of r and p, generate an r-p graph with a growing process.
    """
    for p in [0, 0.25, 0.5, 0.75, 1]:
        for r in [0, 0.25, 0.5, 0.75, 1]:
            fn = '%s/synth_graphs/g-%.2f-%.2f-%.02d.csv' % (data_path, r, p, id)
            print(fn)
            (G, el) = make_rp_graph(fn, m=5, r=r, p=p)
            write_edge_list(el, fn)


if __name__ == '__main__':
    n = 20
    # Run cycles in parallel
    with Pool(processes=n) as pool:
        r = pool.map(do_one_dense_cycle, range(n))
    with Pool(processes=n) as pool:
        r = pool.map(do_one_grow_cycle, range(n))
