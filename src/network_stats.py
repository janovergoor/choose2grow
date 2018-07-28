from collections import Counter
from util import *

"""

  This script contains funtions to compute graph statistics. Most statistics
  are imported from NetworkX, but some are added, and some (like average path
  length) are modified to be computed for a random subset of nodes only.

"""


#
# Individual statistic computation
#

def p_edges(G):
    """
    Compute the share of all possible edges that exist, also known as density.
    All possible edges : n*(n-1)
    Existing edges: e*2
    """
    n = nx.number_of_nodes(G)
    e = nx.number_of_edges(G)
    return e * 2.0 / (n * (n - 1))


def appr_diameter(G, p):
    """
    Faster version of diameter() that only samples a subset of nodes.
    Based on NetworkX code for `diameter()`.
    """
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected.")
    if p == 1.0:
        nodes = G.nodes()
    else:
        n_approx = int(len(G) * p)
        nodes = random.sample(G.nodes(), n_approx)
    D = 0
    for node in nodes:
        path_length = nx.single_source_shortest_path_length(G, node)
        Dt = max(path_length.values())
        if Dt > D:
            D = Dt
    return D


def appr_average_shortest_path_length(G, p):
    """
    Faster version of average_shortest_path_length() that only samples a subset
    of nodes. Based on NetworkX code for `average_shortest_path_length()`.
    Removed the weighted or undirected graph options.
    """
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected.")
    if p == 1.0:
        nodes = G.nodes()
    else:
        n_approx = int(len(G) * p)
        nodes = random.sample(G.nodes(), n_approx)
    avg = 0.0
    for node in nodes:
        path_length = nx.single_source_shortest_path_length(G, node)
        avg += sum(path_length.values())
    n = len(G)
    # re-adjust avg
    avg = avg * (1.0 / p)
    return avg / (n * (n - 1))


def fit_jackson_r(G):
    """
    Fit the plausible r parameter based on the degree distribution.
    Based on the recomended method from Jackson-Rogers [2007].
    """
    degrees = list(dict(G.degree()).values())
    # compute cdf
    counter = Counter(degrees)
    cdf = [0] * max(counter.keys())
    for k, v in counter.items():
        cdf[k - 1] = v
    cdf = np.cumsum(cdf) / float(len(degrees))
    # compute m
    m = np.mean(range(1, len(cdf) + 1))
    # follow Lalo's function
    y = 1 - np.log(1 - cdf + .0001)
    r0 = 1.3
    r1 = 1.7
    delta = abs(r0 - r1)
    k = 1
    while delta > 0.0000001 and k < 50:
        X = np.log(range(1, len(cdf) + 1) + r0 * m)
        A = np.vstack([X, np.zeros(len(X))]).T
        r1 = np.linalg.lstsq(A, y, rcond=None)[0][0]
        delta = abs(r0 - r1)
        r0 = r1
        k = k + 1
    return r1


def fit_r_hat(p_ks, fofs):
    """
    Fit the MLE for \hat{r}, based on a sequence of edges.
    For every edge is recorded:
      p_k - the share of eligible nodes that is a friend of friend of i
      fof - whether the selected node was a friend of friend of i
    """
    def f(r):
        ll = -1
        for (p_k, fof) in zip(p_ks, fofs):
            ll *= fof * (r * p_k + (1 - r)) + (1 - fof) * r * (1 - p_k)
        return ll
    return minimize(f, 0.5, method='nelder-mead', bounds=(0, 1),
                    options={'xtol': 1e-6, 'disp': False})['x'][0]


def fit_exponential(G):
    """
    Fit degree distribution as an exponential distribution,
    return lambda = the mean degree.
    """
    degrees = list(dict(G.degree()).values())
    return 1.0 / np.mean(degrees)


def structural_diversity(G, max_k=5):
    """
    Compute 'structural diversity' for a network. For node, take the ego-graph
    and compute the following metrics:
    - cc1: number of connected compontents
    - cc2: number of 2-connected compontents
    - cc3: number of 3-connected compontents [off]
    - avg_clustering: average clustering coefficient
    - k-core: number of k-cores
    - k-truss: number of k-truss components
    - cliques: number of maximal cliques
    """
    # construct results dictionary
    stats = ['cc1', 'cc2', 'cc3', 'avg_clustering', 'cliques']
    for k in range(1, max_k + 1):
        stats += ['%d-core' % k, '%d-brace' % k]
    res = {x: {} for x in stats}
    # iterate over nodes
    for i in list(G.nodes()):
        # do substracted ego-network stats
        sub_nodes = list(G.neighbors(i))
        H = G.subgraph(sub_nodes)
        # average clustering coefficient
        res['avg_clustering'][i] = nx.average_clustering(H)
        # number of connected components
        res['cc1'][i] = nx.number_connected_components(H)
        # number of bi-connected components
        bicomps = nx.biconnected_components(H)
        # avoid considering dyads as bicomponents
        bicomps = [x for x in bicomps if len(set(x)) > 2]
        res['cc2'][i] = len(bicomps)
        for k in range(1, max_k + 1):
            # do k-cores
            Hkc = kCores(H, k=k)
            res['%d-core' % k][i] = len(list(nx.connected_components(Hkc)))
            # do k-truss
            Hkt = kTruss(Hkc, k=k)
            res['%d-brace' % k][i] = len(list(nx.connected_components(Hkt)))
        # do full ego-network stats
        H = G.subgraph(sub_nodes + [i])
        # number of cliques
        res['cliques'][i] = nx.number_of_cliques(H, i)
    # return result
    return res


def kCores(G_in, k=1):
    """
    Compute k-core subgraph.
    """
    G = G_in.copy()
    # apparently necessary
    G.remove_edges_from(nx.selfloop_edges(G))
    core_numbers = nx.find_cores(G)
    nbunch = [n for n in core_numbers if (core_numbers[n] >= k + 1)]
    G.remove_nodes_from([n for n in G if n not in set(nbunch)])
    return G


def kTruss(G, k=1):
    """
    Compute k-truss on k-core subgraph.
    First, make sure G is a k-core subgraph!
    Adopted from JU's old code
    """
    # calculate embeddedness
    # q - the queue of edges that have been removed, it is used to decrement
    #     the embeddedness of all adjacent edges.
    q = []
    for e in G.edges():
        nodes1 = set(G.neighbors(e[0]))
        nodes2 = set(G.neighbors(e[1]))
        embeddedness = len(nodes1 & nodes2)
        G[e[0]][e[1]]['emb'] = embeddedness
        if (embeddedness < k):
            q.append(e)
    # removal must happen after in order to not distrub iterator
    for e in q:
        G.remove_edge(e[0], e[1])
    while (len(q) != 0):
        e = q.pop()
        nodes1 = set(G.neighbors(e[0]))
        nodes2 = set(G.neighbors(e[1]))
        combined_set = nodes1 & nodes2
        for neighbor in combined_set:
            eprime = (e[0], neighbor)
            G[e[0]][neighbor]['emb'] -= 1
            if (G[e[0]][neighbor]['emb'] < k):
                q.append(eprime)
                G.remove_edge(eprime[0], eprime[1])

            eprime = (e[1], neighbor)
            G[e[1]][neighbor]['emb'] -= 1
            if (G[e[1]][neighbor]['emb'] < k):
                q.append(eprime)
                G.remove_edge(eprime[0], eprime[1])
    q = []
    for (v, d) in G.degree():
        if (d == 0):
            q.append(v)
    G.remove_nodes_from(q)
    return G


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


#
# Stats wrapper functions
#

def process_all_edges(graph, vvv=0):
    """
    Read in a graph from an edge list, compute for every edge the network
    context when that edge was formed, and write out the results.

    'graph' is a filename
    """
    # read the edge list from file
    el = read_edge_list(graph, vvv)
    # extract edge stats from the edge list
    (G, T) = compute_edge_stats(graph, el, n_alt=5, vvv=vvv)
    # write the different outcomes in parts

    # write degrees
    fn = '%s/%s/%s' % (data_path, 'degrees', graph)
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['node', 'degree'])
        for k, v in dict(G.degree()).items():
            writer.writerow([k, v])
    if vvv > 1:
        print("[%s] did degrees" % graph)

    # write stats
    fn = '%s/%s/%s' % (data_path, 'stats', graph)
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['stat', 'value'])
        # compute stats on the outcome graph
        stats = compute_graph_stats(G)
        for k, v in stats.items():
            writer.writerow([k, v])
    if vvv > 1:
        print("[%s] did stats" % graph)

    # write edges
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

    # write (degree) choice sets
    fn = '%s/%s/%s' % (data_path, 'choices', graph)
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['choice_id', 'deg', 'n', 'c'])
        for e in range(len(T)):
            for deg, n in T[e]['degree_distribution'].items():
                c = 1 if deg == T[e]['deg_j'] else 0
                writer.writerow([e, deg, n, c])
    if vvv > 1:
        print("[%s] did choices" % graph)

    # write negatively sampled (degree, fof) choice sets
    fn = '%s/%s/%s' % (data_path, 'choices_sampled', graph)
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['choice_id', 'y', 'deg', 'fof'])
        for e in range(len(T)):
            for (y, deg, fof) in T[e]['mln_data']:
                writer.writerow([e, y, deg, fof])
    if vvv > 0:
        print("[%s] did sampled choices" % graph)


def compute_edge_stats(graph, el, n_alt=5, vvv=0):
    """
    Reconstruct a graph from an edge list and for every edge, compute
    relevant context statistics of the graph when the edge was formed.
    The function returns the final graph and the list of statistics.


    Keyword arguments:

    graph -- an identifier or filename as a string
    el -- edgelist, list of (ts, from, to) tuples
    n_alt -- number of negative choices to sample for sampled choice data
    vvv -- int representing level of debug output [0:none, 1:some, 2:lots]
    """
    # construct 't=0' for actual graphs based on year
    if not synth:
        y = graph.split('_')[-1].split('.')[0]
        t = datetime.strptime(y + '-09-01', "%Y-%m-%d").timestamp()
        for i in range(len(el)):
            if el[i][0] < t:
                el[i][0] = 0
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
    # return graph and edge list
    return (G, T)


def compute_graph_stats(G, p=0.1):
    """
    Wrapper function to compute statistics of a single graph.
    """
    res = {}
    res['n_nodes'] = len(G.nodes())
    res['n_edges'] = len(G.edges())
    res['density'] = nx.density(G)
    res['assortivity'] = nx.degree_assortativity_coefficient(G)
    res['clustering'] = nx.average_clustering(G)
    res['degree_distribution_lambda'] = fit_exponential(G)
    res['fiedler_value'] = nx.algebraic_connectivity(G)
    # TODO - greedy_modularity_communities() from NetworkX is only available in v2.2
    # res['max_modularity'] = modularity(G, greedy_modularity_communities(G))
    res['jackson_r'] = fit_jackson_r(G)
    if nx.is_connected(G):
        res['avg_shortest_path'] = appr_average_shortest_path_length(G, p)
        res['diameter'] = appr_diameter(G, p)
    else:
        Gc = max(nx.connected_component_subgraphs(G), key=len)
        if len(Gc) > len(G) * 0.8:
            res['avg_shortest_path'] = appr_average_shortest_path_length(Gc, p)
            res['diameter'] = appr_diameter(Gc, p)
        else:
            res['avg_shortest_path'] = -1
            res['diameter'] = -1
    return res


def choice_data(id, el, n_alt=5, max_deg=50, vvv=0):
    """
    Short hand function to compute choice set directly from an edge list.
    """
    # extract edge features from edge list
    (G, T) = compute_edge_stats(id, el, n_alt=n_alt, vvv=vvv)
    # extract just the choice set data
    D = []
    for e in range(len(T)):
        for (y, deg, fof) in T[e]['mln_data']:
            D.append([e, y, deg, fof])
    # convert to pandas DataFrame
    D = pd.DataFrame(D, columns=['choice_id','y','deg','fof'])
    # remove too high degree choices
    D = D[D.deg <= max_deg]
    # remove cases without any choice (choice was higher than max_deg)
    D = D[D.groupby('choice_id')['y'].transform(np.sum) == 1]
    return D 

