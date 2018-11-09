import networkx as nx
from multiprocessing import Pool
from util import *

"""

  Driver script #1 - Generate synthetic graphs

  This script generates, for each combination of (r,p), 10 synthetic graphs
  and writes them out as ordered edge lists.
  The structure of the filename is:

      "%s-%.2f-%.2f-%s-%.02d.csv" % ({d,g}, r, p, {u,d}, id)

  output: ../data/synth_graphs

"""

# make sure the output folder exists
mkdir(data_path + '/synth_graphs')

def make_rp_graph(id, G_in=None, n_max=10000, r=0.5, p=0.5, grow=True, m=1, directed=False):
    """
    Generate a graph with n_max edges according to the (r,p)-model, which
    forms edges according to both preferential attachment and triadic closure.

    If grow=True, the graph will be grown (using G_in as seed) as a
    starting graph. For every new node i, m new edges will be generated.
    If grow=False, only new edges will created between existing nodes,
    using the seed graph G_in. Node i of the edge is drawn uniformly at random.
    In either case, the process stops when n_max edges have been formed.

    For every new edge, random sample from two uniform random variables (P, R)
    and create a new edge between i and j, with j selected as follows:
    P<p & R<r - draw j uniformly at random
    P<p & R>r - draw j uniformly at random from the set of i's FoFs
    P>p & R<r - draw j based on j's degree
    P>p & R>r - draw k based on j's degree from the set of i's FoFs

    The function returns an ordered edge list.
    """
    # if no input graph is given, start with a default graph
    if G_in is None:
        if grow:
            # small complete graph for grow
            if directed:
                G = nx.complete_graph(5, create_using=nx.DiGraph())
            else:
                G = nx.complete_graph(5)
        else:
            # sparse ER graph for dense
            G = nx.erdos_renyi_graph(1000, 0.005, directed=directed)
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
            # add j to node sets if it is defined
            if j is not None:
                friends = friends.union(set([j]))
                eligible = eligible - set([j])
                js_friends = G.successors(j) if directed else G.neighbors(j)
                ego2 = ego2.union(js_friends)
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
            if P < p:
                # sample uniformly
                j = random_sample(choice_set)
            else:
                if len(choice_set) == 1:
                    j = list(choice_set)[0]
                else:
                    # sample according to degree
                    ds = G.in_degree(choice_set) if directed else G.degree(choice_set)
                    ds = dict(ds)
                    # add 1 to give new nodes a chance
                    vals = ds.values()
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


def pass_through(graph):
    """
    Make a graph according to the specs in the id string and write it out.
    """
    print(graph)
    (type1, r, p, type2, id) = graph.split('-')
    if type1 not in ['g', 'd'] or type2 not in ['d', 'u']:
        print("[ERROR] id format should be [gd]-%.2f-%.2f-[ud]-.02d")
    (G, el) = make_rp_graph(id, n_max=20000 if type1 == 'g' else 10000,
                            r=float(r), p=float(p),
                            grow=type1 == 'g',
                            m=4 if type1 == 'g' else 1,
                            directed=type2 == 'd')
    fn = '%s/synth_graphs/%s.csv' % (data_path, graph)
    write_edge_list(el, fn)


if __name__ == '__main__':
    n = 10
    todo = []
    for type1 in ['g', 'd']:
        for type2 in ['u', 'd']:
            for p in [0.01, 0.25, 0.5, 0.75, 1.00]:
                for r in [0.01, 0.25, 0.5, 0.75, 1.00]:
                    for id in range(n):
                        fn = '%s-%.2f-%.2f-%s-%.02d' % (type1, r, p, type2, id)
                        todo.append(fn)
    print("TODO: %d" % len(todo))
    # Run cycles in parallel
    with Pool(processes=10) as pool:
        r = pool.map(pass_through, todo)
