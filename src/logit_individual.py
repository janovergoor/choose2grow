from util import *

"""

  This script contains model definitions and objective functions for
  multinomial logit models on *individual* data. The examples are
  represented as (choice_id, Y, degree, n_fofs).

  TODO - incorporate / implement FOF modes

"""


class LogitModel:
    """
    This class represents a generic logit model.
    """
    def __init__(self, model_id, D=None, max_deg=None, vvv=0):
        """
        Constructor for a LogitModel object. The data (the D matrix)
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
        if getattr(self, "grad", None) is not None:
            if self.vvv > 1:
                self.message("fitting with a gradient (using BFGS)")
            # use BFGS if a gradient function is specified
            res = sp.optimize.minimize(lambda x: self.ll(x, w=w), self.u,
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
            res = sp.optimize.minimize(lambda x: self.ll(x, w=w), self.u,
                                       method='L-BFGS-B', callback=print_iter,
                                       bounds=self.bounds,
                                       options={'factr': 1e-12, 'disp': False})
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
    def __init__(self, model_id, D=None, max_deg=50, vvv=False):
        LogitModel.__init__(self, model_id, D, max_deg, vvv)
        self.model_type = 'logit_degree'
        self.model_short = 'd'
        # initate model parameter values
        self.u = [1] * (max_deg + 1)  # current parameter values
        self.se = [None] * (max_deg + 1)  # current SE values

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the degree logit model.
        Computes the likelihood for every data point (choice) separately.

        L(theta, (x,C)) = exp(theta_{k_x}) / sum_{y in C} exp(theta_{k_y})
        """
        # assign exponentiated utilities to all cases
        self.D['score'] = np.exp(u)[self.D.deg]
        # compute total utility per case
        score_tot = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # compute probabilities of choices
        return np.array(self.D.loc[self.D.y == 1, 'score']) / np.array(score_tot)

    def grad(self, u=None, w=None):
        """
        Gradient function of the degree logit model.

        grad_d(theta, D) = sum_{(x,C) in D} [ 1{k_x = d} -
          (sum_{y in C} 1{k_y = d}*exp(theta_k_y)) /
          (sum_{y in C}            exp(theta_k_y))
        ]
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # assign weights to choice sets
        W = pd.DataFrame(data={'choice_id': self.D.choice_id.unique(), 'w': w})
        # assign utilities to all cases
        self.D['score'] = np.exp(u)[self.D.deg]
        # compute total utility per case
        self.D['score_tot'] = self.D.groupby('choice_id')['score'].transform(np.sum)
        # compute probabilities
        self.D['prob'] = self.D['score'] / self.D['score_tot']
        # adjust probabilities based on whether they were chosen
        self.D.loc[self.D.y == 1, 'prob'] -= 1
        # join in weights
        Dt = self.D.merge(W, on='choice_id', how='inner')
        # weight probabilities
        Dt['prob'] *= Dt['w']
        # sum over degrees to get gradient
        return Dt.groupby('deg')['prob'].aggregate(np.sum)


class PolyLogitModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    polynomial functional form: u[i] = sum_d ( i^d * theta[d] )
    """
    def __init__(self, model_id, D=None, max_deg=50, vvv=False, k=2, bounds=None):
        LogitModel.__init__(self, model_id, D, max_deg, vvv)
        self.model_type = 'logit_poly'
        self.model_short = 'p'
        # initate model parameter values
        self.u = [1] * k  # current parameter values
        self.se = [None] * k  # current SE values
        self.bounds = bounds  # bound the parameter

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the polynomial logit model.
        Computes the likelihood for every data point (choice) separately.

        L(theta, (x,C)) = exp(sum_d theta_d*k_x^d) /
                          sum_{y in C} exp(sum_d theta_d*k_y^d))

        However, with k > 2, exp(x^d) gives overflow issues,
        so we use a version of the log-sum-exp trick:

        exp(x) / sum(exp(ys)) = exp(x - max(ys) - log(sum(exp(ys - max(ys)))))

        TODO - can rewrite simpler using sp.misc.logsumexp?
        """
        # raise degree to power
        powers = np.power(np.array([self.D.deg] * len(u)).T, np.arange(len(u)))
        # weight powers by coefficients
        self.D['score'] = np.sum(powers * u, axis=1)
        # compute max score per choice set
        self.D['max_score'] = self.D.groupby('choice_id')['score'].transform(np.max)
        # compute exp of adjusted score
        self.D['score_adj'] = np.exp(self.D.score - self.D.max_score)
        # compute total utility per case
        score_tot = np.log(self.D.groupby('choice_id')['score_adj'].aggregate(np.sum))
        # retrieve max utility (max)
        score_max = self.D.groupby('choice_id')['score'].aggregate(np.max)
        # combine log-sum-exp components
        return np.exp(np.array(self.D.loc[self.D.y == 1, 'score']) - score_max - score_tot)

    def grad(self, u=None, w=None):
        """
        Gradient function of the polynomial logit model.

        grad(theta_d, D) = sum_{(x,C) in D} [ k_x^d -
           (sum_{y in C} k_y^d*exp(sum_d theta_d*k_y^d)) /
           (sum_{y in C}       exp(sum_d theta_d*k_y^d))
        ]

        TODO - implement exp overflow solution from individual_likelihood()
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # raise degree to power
        powers = np.power(np.array([self.D.deg] * len(u)).T, np.arange(len(u)))
        # weight powers by coefficients, exp sum for score
        self.D['score'] = np.exp(np.sum(powers * u, axis=1))
        # initialize empty gradient vector to append to
        grad = np.array([])
        # compute each k-specific gradient separately
        for k in range(len(u)):
            # take degree^k for chosen examples
            choices = self.D.loc[self.D.y == 1, 'deg'] ** k
            # compute 'numerator score'
            self.D['nscore'] = self.D['score'] * (self.D.deg ** k)
            # compute numerator
            num = self.D.groupby('choice_id')['nscore'].aggregate(np.sum)
            # compute denominator
            denom = self.D.groupby('choice_id')['score'].aggregate(np.sum)
            # weight probabilities, add to grad matrix
            grad = np.append(grad, np.sum(w * (np.array(choices) - num/denom)))
        return -1 * grad


class LogLogitModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    log transformation over degrees. The model has 1 parameter.
    """
    def __init__(self, model_id, D=None, max_deg=50, vvv=False, bounds=((0, 2), )):
        LogitModel.__init__(self, model_id, D, max_deg, vvv)
        self.model_type = 'logit_log'
        self.model_short = 'l'
        # initate model parameter values
        self.u = [1]  # current parameter value
        self.se = [None]  # current SE value
        self.bounds = bounds  # bound the parameter
        self.D['log_degree'] = np.log(self.D.deg + log_smooth)  # pre-log degree

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the log logit model.
        Computes the likelihood for every data point (choice) separately.

        L(alpha, (x,C)) = exp(alpha * log(k_x)) / sum_{y in C} exp(alpha * log(k_y))
        """
        # transform degree to score
        self.D['score'] = np.exp(u * np.log(self.D.deg + log_smooth))
        # compute total utility per case
        score_tot = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # compute probabilities of choices
        return np.array(self.D.loc[self.D.y == 1, 'score']) / np.array(score_tot)

    def grad(self, u=None, w=None):
        """
        Gradient function of log logit model.

        grad(alpha, D) = sum_{(x,C) in D} [ alpha*ln(k_x) -
          (sum_{y in C} ln(k_y)*exp(alpha*ln(k_y))) /
          (sum_{y in C}         exp(alpha*ln(k_y)))
        ]
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # transform degree to score
        self.D['score'] = np.exp(u * self.D['log_degree'])
        # take log_degree for chosen examples
        choices = self.D.loc[self.D.y == 1, 'log_degree']
        # compute 'numerator score'
        self.D['nscore'] = self.D['score'] * self.D['log_degree']
        # compute numerator
        num = self.D.groupby('choice_id')['nscore'].aggregate(np.sum)
        # compute denominator
        denom = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # weight probabilities
        return np.array([-1 * np.sum(w * (np.array(choices) - num/denom))])


class MixedLogitModel(LogitModel):
    """
    This class represents a generic mixed logit model. It has similar
    functionality as LogitModel, but the  individual logits that the
    mixed model is composed of have to be added manually.

    The constituent models are represented by the following shortcuts:
    l  = log logit
    px = x-degree poly logit
    d  = degree logit

    TODO - update this whole class
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

    def write_params(self):
        """
        Write out the estimated parameters.
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

