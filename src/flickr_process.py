import networkx as nx
import pandas as pd
from util import *
import csv
import random

"""

  Script to process data for analysis in section 5.2 of the paper.
  It converts the raw "flickr growth" edges to choice data.
  Data is downloadable at:

    http://socialnetworks.mpi-sws.mpg.de/data/flickr-growth.txt.gz

  input : env.data_path/flickr-growth.txt.gz
  output: env.data_path/flickr-growth_choices.csv

"""

# file names
fn_in = data_path + '/flickr-growth.txt.gz'
fn_out = data_path + '/flickr-growth_choices.csv'
url = 'http://socialnetworks.mpi-sws.mpg.de/data'
# date to start sampling from
start_date = ['2006-11-05', '2017-03-01'][0]
# number of choice we want to create
n_sample = 20000
# number of negative samples per choice set
n_alt = 24
# probability that an edge will become a choice set
p = 0.01

# check if the input data has been downloaded yet
if not os.path.exists(fn_in):
    print("[ERROR] Input data not found. Please download with:\n" +
          "        wget %s/flickr-growth.txt.gz %s/ " % (url, data_path))
    exit()

# check if the output data exists already
if os.path.exists(fn_out):
    print("[ERROR] Output data already exists! Please remove it to run.")
    exit()

# read the edge list data
DF = pd.read_csv(fn_in, compression='gzip', header=0, sep='\t',
                 names=['from', 'to', 'ds'])
el_pre = DF[DF.ds < start_date]
el_pre = list(zip(el_pre['from'], el_pre['to']))
el_post = DF[DF.ds >= start_date]
el_post = list(zip(el_post['from'], el_post['to']))
print("Read %d pre rows and %d post rows from %s" %
      (len(el_pre), len(el_post), fn_in))

# create starting graph
G = nx.DiGraph()
G.add_edges_from(el_pre)
print("Starting graph has %d nodes" % len(G))

# open the output file
f_out = open(fn_out, 'wt')
writer = csv.writer(f_out, delimiter=',')
tmp = writer.writerow(['choice_id', 'y', 'deg', 'hops', 'recip', 'n_paths'])


# @profile
def do_edge(i, j, n):
    # get negative samples (few too many)
    js = random.sample(G.nodes, n_alt + 10)
    # WIP optimization
    # X = G.nodes.keys()
    # idx = list((np.random.rand(n_alt + 10) * len(X)).astype(int))
    # js = [X[x] for x in idx]
    # make sure i, j are actually in the graph
    G.add_node(i)
    G.add_node(j)
    # make sure they're not already chosen
    js = [j] + list(set(js) - set(G.successors(i)))[:n_alt]
    # process each option separately
    for new_j in js:
        # compute features
        y = 1 if new_j == j else 0
        try:
            deg = G.in_degree(new_j)
            # sometimes this fails with
            # NetworkXError: nbunch is not a node or a sequence of nodes
            # TypeError: 'numpy.int64' object is not iterable
        except Exception as e:
            print("[ERROR] in (%s,%s) new_j=%s" % (i, j, new_j))
            print(e)
            continue  # just skip it..
        try:
            hops = nx.shortest_path_length(G, source=i, target=new_j)
        except Exception:
            hops = 'NA'
        recip = 1 if G.has_edge(new_j, i) else 0
        if hops == 2:
            try:
                n_paths = nx.all_shortest_paths(G, source=1, target=new_j)
            except Exception:
                n_paths = 0
        else:
            n_paths = 0
        # write out the choice
        tmp = writer.writerow([n, y, deg, hops, recip, n_paths])


# subset rest of the edges
print("Getting %d choice sets" % n_sample)
idx = 0
# go until you have enough
while idx < n_sample:
    (i, j) = el_post.pop(0)
    # only process it if sampled
    if random.random() < p:
        print("Doing edge %d/%d: (%s,%s)" % (idx + 1, n_sample, i, j))
        do_edge(i, j, idx)
        idx += 1
    # actually add the edge
    G.add_edge(i, j)

f_out.close()
print("Wrote %d choice sets to %s" % (idx, fn_out))
