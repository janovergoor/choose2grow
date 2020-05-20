import csv
import networkx as nx
import numpy as np
from glob import glob
from collections import Counter
import plfit
import util
import logit
import synth_generate
import synth_process


def make_figure1_data():
    """
    Figure 1 - Likelihood surface
    """
    D = util.read_data_single("%s/choices/%s.csv" % (util.data_path, 'g-1.00-0.50-u-00'))
    step = 0.01
    scores_uniform = np.array(1.0 / D.groupby('choice_id')['y'].aggregate(len))
    with open("../results/fig1_data.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'p', 'll'])
        for alpha in np.arange(0.0, 2.00, step):
            D['score'] = np.exp(alpha * np.log(D.deg + util.log_smooth))
            score_tot = D.groupby('choice_id')['score'].aggregate(np.sum)
            scores_pa = np.array(D.loc[D.y == 1, 'score']) / np.array(score_tot)
            for p in np.arange(0.0, 1.0, step):
                scores = p * scores_uniform + (1 - p) * scores_pa
                ll = sum(np.log(scores + util.log_smooth))
                x = writer.writerow([alpha, p, ll])
    m = logit.MixedLogitModel('fig1_em', D=D, vvv=2)
    m.add_uniform_model()
    m.add_log_degree_model()
    m.models[1].u[0] = 0.25
    T = m.fit(n_rounds=100, etol=0.001, return_stats=True)
    T.to_csv("../results/fig_1_data_em.csv", index=False)


def make_figure2_data():
    """
    Figure 2 - Attachment function comparing Newman,Pham,degree-model
    """
    (G, el) = synth_generate.make_rp_graph('test', n_max=2000, r=1, p=0.01, directed=False, m=1, grow=True)
    fn = '%s/synth_graphs/test_pa.csv' % util.data_path
    write_edge_list(el, fn)
    synth_process.process_all_edges('test_pa.csv', n_alt=20, vvv=0)
    m = logit.DegreeModel('test_pa.csv', vvv=1, max_deg=100)
    m.fit()
    with open("../results/fig_2_data.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['deg', 'coef', 'se'])
        for i in range(len(m.u)):
            x = writer.writerow([i, m.u[i], m.se[i]])
        m2 = logit.LogDegreeModel('test_pa.csv', vvv=1, max_deg=100)
        m2.fit()
        tmp = writer.writerow(['alpha', m2.u[0], m2.se[0]])


def make_figure3_data():
    """
    Figure 3 - Power law fits on degree of (r,p) graphs
    """
    with open("../results/fig_3_data.csv", 'w') as f:
        writer = csv.writer(f)
        tmp = writer.writerow(['type', 'r', 'p', 'type2', 'id', 'alpha'])
        for graph in glob(util.data_path + '/synth_graphs/g-*-u-*.csv'):
            # read data
            degs = []
            with open(graph, 'r') as f_in:
                reader = csv.reader(f_in)
                tmp = next(reader, None)  # skip the header
                for row in reader:
                    degs += [row[1], row[2]]
            degs = list(Counter(degs).values())
            # run plfit
            alpha = plfit.plfit(degs, discrete=True).plfit()[1]
            # some fits are unstable, skip
            if alpha > 6:
                continue
            # write results
            graph_id = graph[:-4].split('/')[-1].split('-')
            tmp = writer.writerow(graph_id + [alpha])



def make_figure4_data():
    """
    Figure 4 - Log-likelihood of misspecified models
    """
    # Generate left graph
    graph = 'g-0.50-1.00-u-fig3'
    (G, el) = synth_generate.make_rp_graph(id, G_in=nx.complete_graph(10),
                                           n_max=10000, r=0.5, p=1.0,
                                           grow=True, m=5, directed=False)
    write_edge_list(el, '%s/synth_graphs/%s.csv' % (util.data_path, graph))
    synth_process.process_all_edges(graph + '.csv', n_alt=100)
    # Generate right graph
    graph = 'g-1.00-0.50-u-fig3'
    (G, el) = synth_generate.make_rp_graph(id, G_in=nx.complete_graph(10),
                                           n_max=10000, r=1.0, p=0.5,
                                           grow=True, m=5, directed=False)
    write_edge_list(el, '%s/synth_graphs/%s.csv' % (util.data_path, graph))
    synth_process.process_all_edges(graph + '.csv', n_alt=100)
    # read data
    Ds = [util.read_data_single("%s/choices/%s.csv" % (util.data_path, 'g-1.00-0.50-u-fig3')),
          util.read_data_single("%s/choices/%s.csv" % (util.data_path, 'g-0.50-1.00-u-fig3'))]
    titles = ['r=1.00, p=0.50', 'r=0.50, p=1.00']
    xs = np.arange(0, 1.01, 0.01)
    # fit models and write results
    with open("../results/fig_4_data.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['data', 'model', 'p', 'll'])
        for i in range(2):
            # copy model
            m = logit.MixedLogitModel('copy', D=Ds[i], vvv=0)
            m.add_uniform_model()
            m.add_log_degree_model(bounds=((1, 1),))
            for x in xs:
                m.pk = {0: x, 1: 1 - x}
                x = writer.writerow([titles[i], 'p', x, -1 * m.ll()])
            # JR model
            m = logit.MixedLogitModel('jr', D=Ds[i], vvv=0)
            m.add_uniform_model()
            m.add_uniform_fof_model()
            for x in xs:
                m.pk = {0: x, 1: 1 - x}
                x = writer.writerow([titles[i], 'r', x, -1 * m.ll()])


"""
Figure 5 - Non-parametric estimates per model

data processing happens in Rmarkdown reports (../resports/*.Rmd)
"""
