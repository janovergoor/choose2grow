from util import *

"""

  This script contains network generation functions.
  Currently, the only generator is the mixed model, which is described in
  the pydoc of the function itself.

"""


def generate_mixed_model(id, G_in=None, n_max=100, r=0.5, p=0.5, grow=True,
                         m=1, vvv=0):
    """
    Generate a graph with n_max edges according to the full mixed model,
    combining preferential attachment and Jackson-Rogers random connecting
    dynamics.

    If grow=True, the graph will be grown (using G_in as seed) as a
    starting graph. For every new node i, m new edges will be generated.
    If grow=False, only new edges will created between existing nodes,
    using the seed graph G_in. Node i of the edge is drawn uniformly at random.
    In either case, the process stops when n_max edges have been formed.

    For every new edge, random sample from two uniform random variables (P, R)
    and create a new edge between i and j, with j selected as follows:
    P>p & R<r - draw j uniformly at random
    P>p & R>r - draw j uniformly at random from the set of i's FoFs
    P<p & R<r - draw k uniformly at random,
                draw j uniformly at random from k's friends
    P<p & R>r - draw k uniformly at random from i's friends,
                draw j uniformly at random from k's friends

    Intuitively, p determines whether j is drawn uniformly (p=0) or based
    on j's degree (p=1) and r determines whether j is drawn from FoFs (r=0)
    or from the full population (r=1).

    The function returns an ordered edge list.
    """
    # if no input graph is given, start with a default graph
    if G_in is None:
        if grow:
            # small cycle graph for grow
            G = nx.cycle_graph(n=5)
        else:
            # sparse ER graph for dense
            G = nx.erdos_renyi_graph(1000, 0.005)
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
        # create each edge separately
        while m_node < m:
            friends = set(nx.ego_graph(G, i, 1).nodes())
            eligible_js = set(G.nodes()) - friends
            eligible_fofs = set(nx.ego_graph(G, i, 2).nodes()) - friends
            # don't do anything if there are no eligible nodes
            if len(eligible_js) == 0:
                m_node = m
                continue
            # find a node j to connect to
            (P, R) = (random.random(), random.random())
            if R < r or len(friends) < 2 or len(eligible_fofs) == 0:
                if P > p:
                    # random sample from eligible set
                    j = random_sample(eligible_js)
                else:
                    # random sample from full population
                    k = random_sample(G.nodes())
                    # random sample from k's friends
                    j = random_sample(nx.neighbors(G, k))
            else:
                if P < p:
                    # random sample from i's friends
                    k = random_sample(nx.neighbors(G, i))
                    if k is None:
                        j = None
                    else:
                        # random sample from k's friends
                        j = random_sample(nx.neighbors(G, k))
                else:
                    # random sample from i's friend of friends
                    j = random_sample(eligible_fofs)
            # check if edge didn't happen, already exists, or is a self-edge
            if j is None or G.has_edge(i, j) or i == j:
                if vvv > 1:
                    if R < r or len(friends) < 2:
                        if P > p:
                            print("[%s] node:%d - R<r P>p - %d" %
                                  (id, i, len(eligible_js)))
                        else:
                            print("[%s] node:%d - R<r P<p - %d" %
                                  (id, i, len(list(nx.neighbors(G, k)))))
                    else:
                        if P < p:
                            print("[%s] node:%d - R>r P<p - (%d, %d)" %
                                  (id, i, len(list(nx.neighbors(G, i))),
                                  len(list(nx.neighbors(G, k))) if k is not None else 0))
                        else:
                            print("[%s] node:%d - R>r P>p - %d" %
                                  (id, i, len(eligible_fofs)))
                if grow:
                    continue
                else:
                    break
            # add the edge to the edge list and graph
            T.append((m_total, i, j))
            G.add_edge(i, j)
            m_total += 1
            m_node += 1
    if vvv:
        print("[%s] generated a %s graph with %d nodes and %d edges (r=%.2f, p=%.2f)" % \
              (id, 'growing' if grow else 'densifying',
               G.number_of_nodes(), G.number_of_edges(), r, p))
    # return the final graph
    return (G, T)


def generate_configuration_model(G_in):
    """
    Generate a configuration model based on the degree sequence
    of input graph G_in.
    """
    # extract degree sequence
    deg_seq = list(dict(nx.degree(G_in)).values())
    G = nx.configuration_model(deg_seq)
    # remove multi-edges
    G = nx.Graph(G)
    # remove self loops
    G.remove_edges_from(G.selfloop_edges())
    return G
