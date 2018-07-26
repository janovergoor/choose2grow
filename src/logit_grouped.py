from util import *


"""

  This script contains model definitions and objective functions for
  multinomial logit models on *grouped* data. The choice cases are
  represented as columns with the counts of nodes with degree i,
  and the choices are integers representing what the degree of the
  chosen node was.

"""


class LogitModel:
    """
    This class represents a generic logit model.
    """
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=0):
        """
        Constructor for a LogitModel object. The data (the N anc C matrices)
        can be provided directly, or will be read in from file.

        * model_id can be either the file name from where the choices are read,
            or an idenfier for the current model
        * N is a n*d matrix representing the d-dimensional
            options for each of the n examples
        * C is an n*1 vector of choices
        * max_deg is the max degree considered (d-1)
        """
        self.id = model_id
        self.vvv = vvv
        # read data from file if the filename is not specified
        if N is not None and C is not None:
            if N.shape[0] != C.shape[0]:
                self.exception("N and C do not fit together.")
            self.N = N
            self.C = C
        elif '.' in model_id:
            (self.N, self.C) = read_data(model_id, max_deg, vvv=vvv)
        else:
            self.exception("neither filename nor N and C are specified.")
        # initiate the rest of the parameters
        self.n = self.N.shape[0]  # number of examples
        self.d = self.N.shape[1]  # number of degrees considered (max_deg + 1)
        self.n_it = 0  # number of iterations for optimization
        self.bounds = None  # whether there are bounds for the parameters

    def ll(self, u=None, w=None):
        """
        Generic log-likelihood function. It computes individual likelihood
        scores for each example using a model-specifc formula, and computes a
        (weighted) sum of their logs.
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
        scores += log_smooth
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
        # use BFGS when there is a gradient, and no bounds, returns a Hessian
        if getattr(self, "grad", None) is not None and self.bounds is None:
            if self.vvv > 1:
                self.message("fitting with BFGS")
            res = sp.optimize.minimize(lambda x: self.ll(x, w=w), self.u,
                                       jac=lambda x: self.grad(x, w=w),
                                       method='BFGS', callback=print_iter,
                                       options={'gtol': 1e-8})
            # store the standard errors
            H = res.hess_inv
            H = np.diag(np.diagonal(np.linalg.inv(H)))
            self.se = np.diagonal(np.sqrt(np.linalg.inv(H)))
        # else, use L-BFGS-B
        else:
            if self.vvv > 1:
                self.message("fitting with L-BFGS-B")
            res = sp.optimize.minimize(lambda x: self.ll(x, w=w), self.u,
                                       method='L-BFGS-B', callback=print_iter,
                                       bounds=self.bounds)
        # store the resulting parameters
        self.u = res.x

    def write_params(self):
        """
        Write out the estimated parameters as csv.
        Colums are: ['id', 'mode', 'p(k)', 'parameter', 'value', 'se']
        """
        if self.id is None:
            self.exception("can't write model, as filename not specified.")
        if '.' not in self.id:
            self.exception("model id is not a filename.")
        # make sure the output folder exists
        folder = '%s/fits/%s' % (data_path, self.model_type)
        mkdir(folder)
        with open('%s/%s' % (folder, self.id), 'w') as f:
            writer = csv.writer(f)
            # write each degree as a row
            for i in range(len(self.u)):
                row = [self.id,
                       self.model_type,
                       1,
                       i,
                       self.u[i],
                       self.se[i]]
                writer.writerow(row)
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
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=False):
        LogitModel.__init__(self, model_id, N, C, max_deg, vvv)
        self.model_type = 'logit_degree'
        self.model_short = 'd'
        # initate model parameter values
        self.u = [1] * self.d  # current parameter values
        self.se = [None] * self.d  # current SE values

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the grouped degree logit model.
        Computes the likelihood for every data point (choice) separately.

        L(theta, (x,C)) = exp(theta_{k_x}) / sum_{j in 1:K} n_j * exp(theta_j)
        """
        # compute exponentiated utilities
        score = np.array([np.exp(u)] * self.n)
        # compute total utility per case
        score_tot = np.sum(self.N * score, axis=1)  # row sum
        # compute probabilities of choices
        return score[range(self.n), self.C] / score_tot

    def grad(self, u=None, w=None):
        """
        Gradient function of the grouped degree logit model.

        grad_d(theta, D) = sum_{(x,C) in D} [ 1{k_x = d} -
          n_d * exp(theta_d) / (sum_{j in 1:K} n_j * exp(theta_j)) ]
        """
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # make weight matrix
        w = np.array([w] * self.d).T
        # compute exponentiated utilities
        E = np.array([np.exp(u)] * self.n)
        # assign utilities to all cases
        score = self.N * E
        # compute total utility per case
        score_tot = np.sum(score, axis=1)  # row sum
        # compute probabilities
        prob = (score.T / score_tot).T
        # subtract 1 from chosen degree
        prob[range(self.n), self.C] = prob[range(self.n), self.C] - 1
        # sum over degrees to compute gradient
        return np.sum(prob * w, axis=0)  # col sum

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
        """
        Individual likelihood function of the grouped polynomial logit model.
        Computes the likelihood for every data point (choice) separately.

        L(theta, (x,C)) = exp(sum_d theta_d * k_x^d) /
                          sum_{j in 1:K} n_j * exp(sum_d theta_d * j^d))
        
        However, with k > 2, exp(x^d) gives overflow issues,
        so we use a version of the log-sum-exp trick:

        exp(x) / sum(ns * exp(ys)) = exp(x - log(sum(ns * exp(ys))))
        """
        # compute poly utilities, repeat for every instance
        score = np.array([poly_utilities(self.d, u)] * self.n)
        # combine log-sum-exp components
        return np.exp(score[range(self.n), self.C] - sp.misc.logsumexp(score, axis=1, b=self.N))

    def grad(self, u=None, w=None):
        """
        Gradient function of the grouped polynomial logit model.

        grad(theta_d, D) = sum_{(x,C) in D} [ k_x^d -
           (sum_{j in 1:K} j^d * n_j * exp(sum_d theta_d * j^d)) /
           (sum_{j in 1:K}       n_j * exp(sum_d theta_d * j^d))
        ]

        Here, we also use the log-sum-exp trick in the fraction.

        TODO - 129% discrepancy with ll(), but still fits seemingly correctly...:

            >>> from logit_grouped import *
            >>> m1 = PolyLogitModel('d-0.75-0.50-00.csv', k=2)
            >>> check_grad_rel(m1.ll, m1.grad, m1.u)
            array([ 0.        , -1.29782042])
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # compute poly utilities, repeat for every instance, exponentiate
        score = np.array([poly_utilities(self.d, u)] * self.n)
        # make matrix of degrees to take power
        D = np.array([range(self.d)] * self.n)
        # initialize empty gradient vector to append to
        grad = np.array([])
        # compute gradient for every polynomial degree separately
        for k in range(len(u)):
            # compute 'numerator': power degree * group n * poly utility
            num = sp.misc.logsumexp(score, axis=1, b=np.power(D, k) * self.N)
            # compute 'denominator': group n * poly utility
            denom = sp.misc.logsumexp(score, axis=1, b=self.N)
            # score is degree choice ^ poly degree, normalized by division
            scores = self.C ** k - np.exp(num - denom)
            # sum over rows, potentially weighted
            grad = np.append(grad, np.sum(scores * w))
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
    """
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=False, bounds=None):
        LogitModel.__init__(self, model_id, N, C, max_deg, vvv)
        self.model_type = 'logit_log'
        self.model_short = 'l'
        # initate model parameter values
        self.u = [1]  # current parameter value
        self.se = [None]  # current SE value
        self.bounds = bounds  # bound the parameter

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the grouped log logit model.
        Computes the likelihood for every data point (choice) separately.

        L(alpha, (x,C)) = exp(alpha * log(k_x)) /
                          sum_{j in 1:K} n_j * exp(alpha * log(j))

        """
        # compute log utility per degree
        score = np.exp(u * np.log(np.array([range(self.d)] * self.n) + log_smooth))
        # compute total utility per case
        score_tot = np.sum(self.N * score, axis=1)  # row sum
        # compute probabilities of choices
        return score[range(self.n), self.C] / score_tot

    def grad(self, u=None, w=None):
        """
        Gradient function of the grouped log logit model.

        grad(alpha, D) = sum_{(x,C) in D} [ ln(k_x) -
          (sum_{j in 1:K} ln(j) * n_j * exp(alpha*ln(j))) /
          (sum_{j in 1:K}         n_j * exp(alpha*ln(j)))
        ]
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # make matrix of log degree
        D = np.log(np.array([range(self.d)] * self.n) + log_smooth)
        # compute numerator : log degree * group n * log utility
        num = np.sum(D * self.N * np.exp(u * D), axis=1)  # row sum
        # compute denominator : group n * log utility
        denom = np.sum(self.N * np.exp(u * D), axis=1)  # row sum
        # normalize by total summed up exponentiated utility
        scores = D[range(self.n), self.C] - num / denom
        # sum over rows, potentially weighted
        return -1 * np.array([np.sum(scores * w)])


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

    def add_log_model(self, bounds=None):
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

    def write_params(self):
        """
        Write out the estimated parameters as csv.
        Colums are: ['id', 'mode', 'p(k)', 'parameter', 'value', 'se']
        """
        if self.id is None:
            self.exception("can't write model, as filename not specified.")
        if '.' not in self.id:
            self.exception("model id is not a filename.")
        # make sure the output folder exists
        folder = '%s/fits/%s' % (data_path, self.model_type)
        mkdir(folder)
        # construct the output file handle
        fn = self.make_fn()
        with open('%s/%s' % (folder, fn), 'w') as f:
            writer = csv.writer(f)
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
        if self.vvv:
            self.message("wrote model to file")

    def make_fn(self):
        """
        Construct a filename, including contituent model shorts.
        """

        fn = self.id.replace('.csv', '')
        fn = '%s-%s.csv' % (fn, self.model_short)
        return fn


def read_data_single(fn, max_deg):
    """
    Read data (options and choices) for a single graph.
    If the max observed degree for a graph is less than
    max_deg, fill it in with zeros.
    """
    path = '%s/%s/%s' % (data_path, 'choices', fn)
    # read the choices
    dg = pd.read_csv(path)
    # remove too high degree choices
    dg = dg[dg.deg <= max_deg]
    # remove cases without any choice (choice was higher than max_deg)
    dg = dg[dg.groupby('choice_id')['c'].transform(np.sum) == 1]
    # convert counts to matrix
    ids = sorted(list(set(dg['choice_id'])))  # unique choices
    did = dict([(ids[x], x) for x in range(len(ids))])  # dictionary
    xs = [did[x] for x in dg.choice_id]  # converted indices
    # construct the matrix
    N = np.zeros((len(ids), max_deg + 1))
    N[xs, dg.deg] = dg.n
    # convert choices to vector
    C = np.array(dg[dg.c == 1].deg)
    return (N, C)


def read_data(fn, max_deg=50, vvv=False):
    """
    Read data for either a single graph, or all graphs with the
    specified parameters. Degrees are cut-off at max_deg.
    """
    if 'all' in fn:
        # read all
        Ns = []
        Cs = []
        # get all files that match
        fn_tmp = '-'.join(fn.split('-')[:-1])
        pattern = "%s/choices/%s*.csv" % (data_path, fn_tmp)
        fns = [os.path.basename(x) for x in glob(pattern)]
        for x in fns:
            (N, C) = read_data_single(x, max_deg)
            Ns.append(N)
            Cs.append(C)
        # append the results
        N = np.vstack(Ns)
        C = np.hstack(Cs)
    else:
        # read one
        (N, C) = read_data_single(fn, max_deg)
    # cut off at max observed degree
    md = np.max(np.arange(max_deg + 1)[np.sum(N, axis=0) > 0])
    N = N[:, :(md + 1)]
    if vvv:
        print("[%s] read (%d x %d)" % (fn, N.shape[0], N.shape[1]))
    return (N, C)


def poly_utilities(n, theta):
    """
    Generate utilities of the form: u[i] = sum_k (i^k * theta[i])
    """
    u = np.array([0.0] * n)
    for i in range(len(theta)):
        u += np.array(range(n))**i * theta[i]
    return u
