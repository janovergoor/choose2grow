from util import *

"""

  This script contains model definitions and objective functions for
  multinomial logit models on *individual* data. The examples are
  represented as (choice_id, Y, degree, n_fofs).

  TODO - this whole script is a WIP draft

"""


class LogitModel:
    """
    This class represents a generic logit model.
    """
    def __init__(self, model_id, D=None, max_deg=None, vvv=0):
        """
        Constructor for a LogitModel object. The data (the N anc C matrices)
        can be provided directly, or will be read in from file.

        * model_id can be either the file name from where the choices are read,
            or an idenfier for the current model
        * D is a (n*6)x4 matrix, where each case has 6 rows (1 pos, 5 neg),
            and for every example we get [choice_id, Y, degree, n_fofs]
        """
        self.id = model_id
        self.vvv = vvv
        # read data from file if the filename is not specified
        if D is not None:
            self.D = D
        elif '.' in model_id:
            self.D = read_data(model_id, max_deg=max_deg, vvv=vvv)
        else:
            self.exception("neither filename nor D are specified.")
        # initiate the rest of the parameters
        self.n = len(set(self.D.choice_id))  # number of examples
        self.n_it = 0  # number of iterations for optimization
        self.bounds = None  # whether there are bounds for the parameters

    def ll(self, u=None, w=None):
        """
        Generic log-likelihood function. It computes individual log-likelihood
        scores for each example using a model-specifc formula, and computes a
        (weighted) sum.
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # compute individual likelihood scores
        scores = self.individual_likelihood(u)
        # add tiny smoothing for deg=0 choices
        scores += 0.00001
        # return sum
        return -1 * sum(np.log(scores) * w)

    def fit(self, w=None):
        """
        Fit the model using scipy.optimize.minimize()
        """
        # reset number of iterations
        self.n_it = 0

        # helper function to print out intermediate values of the ll function
        def print_iter(x):
            if self.n_it % 10 == 0 and self.vvv > 1:
                self.message("i=%3d ll=%.5f" % (self.n_it, self.ll(x)))
            self.n_it += 1

        # check what method to use
        if getattr(self, "grad", None) is not None:
            if self.vvv > 1:
                self.message("fitting with a gradient (using BFGS)")
            # use BFGS if a gradient function is specified
            res = minimize(lambda x: self.ll(x, w=w), self.u,
                           jac=lambda x: self.grad(x, w=w),
                           method='BFGS', callback=print_iter,
                           options={'gtol': 1e-8, 'disp': False})
            # store the standard errors
            H = res.hess_inv
            H = np.diag(np.diagonal(np.linalg.inv(H)))
            self.se = np.diagonal(np.sqrt(np.linalg.inv(H)))
        else:
            if self.vvv > 1:
                self.message("fitting without a gradient (using L-BFGS-B)")
            # else, use L-BFGS-B
            res = minimize(lambda x: self.ll(x, w=w), self.u,
                           method='L-BFGS-B', callback=print_iter,
                           bounds=self.bounds,
                           options={'factr': 1e-12, 'disp': False})
        # store the resulting parameters
        self.u = res.x

    def write_degree_utilities(self):
        """
        Write out the estimated utilities of every degree from 1:max_degree.
        """
        if self.id is None:
            self.exception("can't write model, as filename not specified.")
        if '.' not in self.id:
            self.exception("model id is not a filename.")
        # make sure the output folder exists
        folder = '%s/fits_utility/%s' % (data_path, self.model_type)
        mkdir(folder)
        # construct the output file handle
        f = open('%s/%s' % (folder, self.id), 'w')
        writer = csv.writer(f)
        # write the header
        if self.se[0] is not None:
            # write out confidence intervals if SE is specified
            header = ['deg', 'u', 'll', 'ul', 'u_exp', 'll_exp', 'ul_exp']
        else:
            header = ['deg', 'u', 'u_exp']
        writer.writerow(header)
        # write each degree as a row
        for d in range(self.d):
            if self.se[0] is not None:
                row = [d,
                       self.predict(d),
                       self.predict(d) - 1.96 * self.se[d],
                       self.predict(d) + 1.96 * self.se[d],
                       np.exp(self.predict(d)),
                       np.exp(self.predict(d) - 1.96 * self.se[d]),
                       np.exp(self.predict(d) + 1.96 * self.se[d])]
            else:
                row = [d,
                       self.predict(d),
                       np.exp(self.predict(d))]
            writer.writerow(row)
        # close the file
        f.close()
        if self.vvv:
            self.message("wrote model to file")

    def write_params(self):
        """
        Write out the estimated parameters.
        """
        if self.id is None:
            self.exception("can't write model, as filename not specified.")
        if '.' not in self.id:
            self.exception("model id is not a filename.")
        # make sure the output folder exists
        folder = '%s/fits_param/%s' % (data_path, self.model_type)
        mkdir(folder)
        # construct the output file handle
        f = open('%s/%s' % (folder, self.id), 'w')
        writer = csv.writer(f)
        # write the header
        header = ['id', 'mode', 'pk', 'param', 'value', 'se']
        writer.writerow(header)
        # write each degree as a row
        for i in range(len(self.u)):
            row = [self.id,
                   self.model_type,
                   1,
                   i,
                   self.u[i],
                   self.se[i]]
            writer.writerow(row)
        # close the file
        f.close()
        if self.vvv:
            self.message("wrote model to file")

    def message(self, s):
        """
        Generic message wrapper.
        """
        print("[%s] %s" % (self.id, s))

    def exception(self, s):
        """
        Generic exception wrapper.
        """
        raise Exception("[%s] %s" % (self.id, s))


class DegreeLogitModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    distinct coefficient beta_i for each individual degree i.
    """
    def __init__(self, model_id, D=None, max_deg=50, vvv=False):
        LogitModel.__init__(self, model_id, D, max_deg, vvv)
        self.model_type = 'logit_degree'
        self.model_short = 'd'
        # initate model parameter values
        self.u = [1] * (max_deg + 1)  # current parameter values
        self.se = [None] * (max_deg + 1)  # current SE values

    def individual_likelihood(self, u):
        # compute exponentiated utilities
        E = np.array([np.exp(u)] * self.D.shape[0])
        D = self.D
        # assign utilities to all cases
        D['score'] = E[range(E.shape[0]), D.deg]
        # compute total utility per case
        D['score_tot'] = D.groupby('choice_id')['score'].transform(np.sum)
        # only look at selected options only
        D = D[D.y == 1]
        # compute probabilities of choices
        return D.score / D.score_tot

    def grad(self, u=None, w=None):
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # make weights
        w = np.array([w] * len(self.u)).T
        # compute exponentiated utilities

        # TODO
        E = np.array([np.exp(u)] * self.n)
        E = np.array([np.exp(u)] * self.D.shape[0])
        # assign utilities to all cases
        score = self.N * E
        # compute total utility per case
        score_tot = np.sum(score, axis=1)  # row sum
        # compute probabilities
        prob = (score.T / score_tot).T
        # adjust probabilities based on whether they were chosen
        prob[range(self.n), self.C] = prob[range(self.n), self.C] - 1
        # (col) sum over degrees to compute gradient
        return np.sum(prob * w, axis=0)

    def predict(self, d):
        """
        Return utility of degree d according to fitted model.
        """
        if d > self.d:
            self.exception("asked for utility exceeds max degree")
        return self.u[d]


class PolyLogitModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    polynomial functional form: [i] = sum_k ( i^k * theta[i] )
    """
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=False,
                 k=2, bounds=None):
        LogitModel.__init__(self, model_id, N, C, max_deg, vvv)
        self.model_type = 'logit_poly'
        self.model_short = 'p'
        # initate model parameter values
        self.u = [1] * k  # current parameter values
        self.se = [None] * k  # current SE values
        self.bounds = bounds  # bound the parameter

    def individual_likelihood(self, u):
        # compute poly utilities
        E = np.array([poly_utilities(self.d, u)] * self.n)
        # assign utilities to all cases
        score = self.N * E
        # compute total utility per case
        score_tot = np.sum(score, axis=1)  # row sum
        # compute probabilities of choices
        return score[range(self.n), self.C] / score_tot

    def grad_wip(self, u=None, w=None):
        """
        Gradient of the log logit model:
        grad(beta_k) = sum_i [ u_i^k - (sum_j exp(sum_k beta_k u_i^k) * u_i^k) / (sum_j exp(sum_k beta_k u_j^k) ]
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # compute poly utilities
        E1 = np.exp(np.array([poly_utilities(self.d, u)] * self.n))
        # compute matrix of degrees
        E2 = np.array([range(self.d)] * self.n)
        # initialize empty gradient vector to append to
        grad = np.array([])
        for k in range(len(u)):
            # compute numerator (utility * power degree * n)
            num = np.sum(E1 * np.power(E2, k) * self.N, axis=1)
            # normalize by total summed up exponentiated utility
            p = self.C ** k - (num / np.sum(E1 * self.N, axis=1))
            # sum over rows, potentially weighted
            grad = np.append(grad, np.sum(p * w))
        return -1 * grad

    def predict(self, d):
        """
        Return utility of degree d according to fitted model.
        """
        return poly_utilities(d + 1, self.u)[d]


class LogLogitModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    log transformation over degrees. The model has 1 parameter.
    p(deg_i) = exp(u * log(deg_i)) * n_i / sum_j exp(u * log(deg_j)) n_j
    p(x_i  ) = exp(u * log(deg_i))       / sum_j exp(u * log(deg_j))
    TODO - which one?
    """
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=False,
                 bounds=((0, 2), )):
        LogitModel.__init__(self, model_id, N, C, max_deg, vvv)
        self.model_type = 'logit_log'
        self.model_short = 'l'
        # initate model parameter values
        self.u = [1]  # current parameter value
        self.se = [None]  # current SE value
        self.bounds = bounds  # bound the parameter

    def individual_likelihood(self, u):
        # compute matrix of degrees
        E = np.array([range(self.d)] * self.n)
        # set deg 0 to 1 for now...
        E[:, 0] = 1
        # compute exponentiated utilities of LOG degrees
        E = np.exp(u * np.log(E))
        # set deg 0 to 0 again
        E[:, 0] = 0
        # assign utilities to all cases
        score = self.N * E
        # compute total utility per case
        score_tot = np.sum(score, axis=1)  # row sum
        # compute probabilities of choices
        return score[range(self.n), self.C] / score_tot  # P(deg_i)
        # return np.exp(u * np.log(self.C)) / score_tot  # P(x_i)

    def grad_wip(self, u=None, w=None):
        """
        Gradient of the log logit model:
        grad(a) = sum_i [ log(u_i) - (sum_j u_j^a * log(u_j)) / sum_j u_j^a ]
        TODO - for some values it doesn't work..
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # construct matrix of degrees
        E = np.array([range(self.d)] * self.n)
        # raise degrees to the power of u, multiply by number of options
        P = np.power(E, u) * self.N
        # set deg 0 to 1 for now...
        E[:, 0] = 1
        # initiate zero matrix
        X = np.zeros(self.N.shape)
        # normalize row sums by log(deg)
        # ugly hack to avoid log(0)
        X[self.N > 0] = P[self.N > 0] * np.log(E[self.N > 0])
        # revert back to 0 (probably not necessary)
        X[:, 0] = 0
        # sum up log(deg)-normalized numerator
        num = np.sum(X, axis=1)  # row sum
        # compute individual row values
        p = np.log(self.C) - (num / np.sum(P, axis=1))  # inverse?
        # sum over rows, potentially weighted
        return np.array([-1 * np.sum(p * w)])

    def predict(self, d):
        """
        Return utility of degree d according to fitted model.
        """
        return np.log(d) * self.u[0]


class MixedLogitModel(LogitModel):
    """
    This class represents a generic mixed logit model. It has similar
    functionality as LogitModel, but the  individual logits that the
    mixed model is composed of have to be added manually.

    The constituent models are represented by the following shortcuts:
    l  = log logit
    px = x-degree poly logit
    d  = degree logit
    """
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=False):
        LogitModel.__init__(self, model_id, N, C, max_deg, vvv)
        self.model_type = 'mixed_logit'
        self.max_deg = max_deg
        self.vvv = vvv
        self.models = []  # list of models
        self.pk = {}  # class probabilities
        self.model_short = ''

    def add_degree_model(self):
        """
        Add a degree logit model to the list of models.
        """
        self.models += [DegreeLogitModel(self.id, N=self.N, C=self.C,
                                         max_deg=self.max_deg,
                                         vvv=self.vvv)]
        self.model_short += 'd'

    def add_log_model(self, bounds=((0, 2), )):
        """
        Add a log logit model to the list of models.
        """
        self.models += [LogLogitModel(self.id, N=self.N, C=self.C,
                                      max_deg=self.max_deg, vvv=self.vvv,
                                      bounds=bounds)]
        self.model_short += 'l'

    def add_poly_model(self, k=2, bounds=None):
        """
        Add a poly logit model to the list of models.
        """
        self.models += [PolyLogitModel(self.id, N=self.N, C=self.C,
                                       max_deg=self.max_deg, vvv=self.vvv,
                                       k=k, bounds=bounds)]
        self.model_short += 'p%d' % k

    def fit(self, n_rounds=20, etol=0.01):
        """
        Fit the mixed model using a version of EM.
        """
        ms = self.models  # shorthand
        K = len(ms)
        if K < 1:
            self.exception("not enough models specified for mixed model")
        # initate class probabilities
        self.pk = {k: 1.0 / K for k in range(K)}
        # store previous LL
        prev_ll = {k: 10e10 for k in range(K)}
        # run EM n_rounds times
        for i in range(n_rounds):
            if self.vvv:
                self.message("Round %d/%d" % (i + 1, n_rounds))
            # 1) Expectation - find class responsibilities given weights
            # compute probabilities for individual examples
            probs = {k: ms[k].individual_likelihood(ms[k].u) for k in range(K)}
            # compute numerator (for each individual, the sum of likelihoods)
            num = [np.sum([self.pk[k] * probs[k][j] for k in range(K)]) for j in range(self.n)]
            # compute responsibilities by normalizing w total class probability
            gamma = {k: (self.pk[k] * probs[k]) / num for k in range(K)}
            # 2) Maximization - find optimal coefficients given current weights
            for k in range(K):
                # update average class responsibilities
                self.pk[k] = np.mean(gamma[k])
                # actually run the optimizer for current class
                ms[k].fit(w=gamma[k])
            if self.vvv:
                for i in range(K):
                    self.message("[%d:%12s] P(class)=%.5f  ll=%.4f  u=%s" %
                                 (i, ms[i].model_type, self.pk[i],
                                  ms[i].ll(w=gamma[i]), ms[i].u))
            # compute current total likelihood
            ll = {k: ms[k].ll(w=gamma[k]) for k in range(K)}
            delta = np.sum([abs(prev_ll[k] - ll[k]) for k in range(K)])
            # stop if little difference
            if delta < etol:
                self.message("delta in ll < %f, stopping early" % etol)
                break
            # store new ll
            prev_ll = ll

    def write_degree_utilities(self):
        """
        Write out a fitted mixed model.
        """
        if self.id is None:
            self.exception("can't write model, as filename not specified.")
        if '.' not in self.id:
            self.exception("model id is not a filename.")
        # make sure the output folder exists
        folder = '%s/fits_utility/%s' % (data_path, self.model_type)
        mkdir(folder)
        # construct the output file handle
        f = open('%s/%s' % (folder, self.id), 'w')
        writer = csv.writer(f)
        # write the header
        writer.writerow(['model', 'pk', 'deg', 'u', 'u_exp'])
        # write each model
        for k in range(len(self.models)):
            # write each degree to compare across models
            for d in range(self.models[k].d):
                row = [self.models[k].model_type,
                       "%.3f" % self.pk[k],
                       d + 1,
                       self.models[k].predict(d),
                       np.exp(self.models[k].predict(d))]
                writer.writerow(row)
        # close the file
        f.close()
        if self.vvv:
            self.message("wrote model to file")

    def write_params(self):
        """
        Write out the estimated parameters.
        """
        if self.id is None:
            self.exception("can't write model, as filename not specified.")
        if '.' not in self.id:
            self.exception("model id is not a filename.")
        # make sure the output folder exists
        folder = '%s/fits_param/%s' % (data_path, self.model_type)
        mkdir(folder)
        # construct the output file handle
        fn = self.make_fn()
        f = open('%s/%s' % (folder, fn), 'w')
        writer = csv.writer(f)
        # write the header
        header = ['id', 'mode', 'pk', 'param', 'value', 'se']
        writer.writerow(header)
        # write each model
        for k in range(len(self.models)):
            # grab model
            m = self.models[k]
            # write each degree as a row
            for i in range(len(m.u)):
                row = [fn,
                       m.model_type,
                       "%.3f" % self.pk[k],
                       i,
                       m.u[i],
                       m.se[i]]
                writer.writerow(row)
        # close the file
        f.close()
        if self.vvv:
            self.message("wrote model to file")

    def make_fn(self):
        """
        Construct a filename, including contituent model shorts.
        """

        fn = self.id.replace('.csv', '')
        fn = '%s-%s.csv' % (fn, self.model_short)
        return fn


def read_data_single(fn, max_deg=None):
    """
    Read data (options and choices) for a single graph.
    """
    path = '%s/%s/%s' % (data_path, 'choices_sampled', fn)
    # read the choices
    D = pd.read_csv(path)
    if max_deg is not None:
        # remove too high degree choices
        D = D[D.deg <= max_deg]
        # remove cases without any choice (choice was higher than max_deg)
        D = D[D.groupby('choice_id')['y'].transform(np.sum) == 1]
    # read the choices
    return D


def read_data(fn, max_deg=None, vvv=False):
    """
    Read data for either a single graph, or all graphs with the
    specified parameters.
    """
    if 'all' in fn:
        # read all
        Ds = []
        # get all files that match
        fn_tmp = '-'.join(fn.split('-')[:-1])
        pattern = "%s/choices_sampled/%s*.csv" % (data_path, fn_tmp)
        fns = [os.path.basename(x) for x in glob(pattern)]
        for x in fns:
            D = read_data_single(x, max_deg)
            # update choice ids so they dont overlap
            fid = x.split('.csv')[0].split('-')[-1]
            ids = [('%09d' + fid) % x for x in D.choice_id]
            D.choice_id = ids
            Ds.append(D)
        # append the results
        D = np.hstack(Ds)
    else:
        # read one
        D = read_data_single(fn, max_deg)
    # cut off at max observed degree
    if vvv:
        print("[%s] read (%d x %d)" % (fn, D.shape[0], D.shape[1]))
    return D


def poly_utilities(n, theta):
    """
    Generate utilities of the form: u[i] = sum_k (i^k * theta[i])
    """
    u = np.array([0.0] * n)
    for i in range(len(theta)):
        u += np.array(range(n))**i * theta[i]
    return u
