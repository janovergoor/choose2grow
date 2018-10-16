import csv
import numpy as np
import util
import logit
import synth_generate
import synth_process

# Figure 1 - log-likelihood surface
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


# Figure 2 - "PA vs Pham"
(G, el) = synth_generate.make_rp_graph('test', n_max=10000, r=1, p=0.01, directed=False, m=1, grow=True)
fn = '%s/synth_graphs/test_pa.csv' % util.data_path
synth_generate.write_edge_list(el, fn)
synth_process.process_all_edges('test_pa.csv', n_alt=20, vvv=0)
m = logit.DegreeModel('test_pa.csv', vvv=1, max_deg=100)
m.fit()
with open("../results/fig2_deg_fit.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['deg', 'coef', 'se'])
    for i in range(len(m.u)):
        x = writer.writerow([i, m.u[i], m.se[i]])
    m2 = logit.LogDegreeModel('test_pa.csv', vvv=1, max_deg=100)
    m2.fit()
    x = writer.writerow(['alpha', m2.u[0], m2.se[0]])
