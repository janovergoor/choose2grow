import csv
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import util


"""

  This script contains the generic LogitModel and MixedLogit classes.
  These both have data loading, model fitting, and output functionality.
  It also includes model definitions and objective functions for individual modes.
  The examples (data) are represented as (choice_id, Y, degree, n_fofs).

"""

class LogitModel:
    """
    This class represents a generic logit model.
    """
    def __init__(self, model_id, max_deg=50, bounds=None, D=None, vvv=0):
        """
        Constructor for a LogitModel object. The data can be provided directly,
        or it can be read in from file.

        Keyword arguments:

        model_id -- model_id can be either the file name from where the choices
            are read, or an idenfier for the current model
        max_deg -- the max degree that will be considered (default: 50)
        D -- 2d-array representing choice options
        vvv -- int representing level of debug output [0:none, 1:some, 2:lots]

        If data is supplied directly, D is a (n*i)x4 matrix, where each
        choice set has i choices, exactly one of which should be chosen.
        For every example we get the following covariates:
           [choice_id, Y, degree, n_fofs]
        """
        self.id = model_id
        self.vvv = vvv

        if D is not None:
            self.D = D
        elif '.' in model_id:
            self.D = util.read_data(model_id, max_deg, vvv=vvv)
        else:
            self.exception("neither filename nor D are specified..")
        self.n = len(set(self.D.choice_id))  # number of examples
        self.d = max_deg + 1  # number of degrees considered

        # initiate the rest of the parameters
        self.n_it = 0  # number of iterations for optimization
        self.u = [1]  # current parameter value
        self.se = [None]  # current SE value
        self.bounds = bounds  # whether there are bounds for the parameters

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
        scores += util.log_smooth
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
            res = minimize(lambda x: self.ll(x, w=w), self.u,
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
            res = minimize(lambda x: self.ll(x, w=w), self.u,
                           method='L-BFGS-B', callback=print_iter,
                           bounds=self.bounds)
        # store the resulting parameters
        self.u = res.x
        if self.vvv > 0:
            self.message("parameters after fitting: " + str(self.u))

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
        folder = '%s/fits/%s' % (util.data_path, self.model_type)
        util.mkdir(folder)
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


class MixedLogitModel(LogitModel):
    """
    This class represents a generic mixed logit model. It has similar
    functionality as LogitModel, but the individual logits that the mixed model
    is composed of have to be added manually.

    The constituent models are represented by the following shortcuts:
      l  = log logit
      px = x-degree poly logit
      d  = degree logit
    """
    def __init__(self, model_id, max_deg=50, D=None, vvv=0):
        """
        Constructor for a MixedLogitModel object. It inherits from LogitModel,
        but is also composed of LogitModel modes. Like LogitModel, the data can
        be provided directly, or it can be read in from file.

        Keyword arguments:

        model_id -- model_id can be either the file name from where the choices
            are read, or an idenfier for the current model
        max_deg -- the max degree that will be considered (default: 50)
        D -- 2d-array representing choice options
        vvv -- int representing level of debug output [0:none, 1:some, 2:lots]
        """
        LogitModel.__init__(self, model_id, max_deg=max_deg, D=D, vvv=vvv)
        self.model_type = 'mixed_logit'
        self.max_deg = max_deg
        self.vvv = vvv
        self.models = []  # list of models
        self.pk = {}  # class probabilities
        self.model_short = ''

    def add_uniform_model(self):
        """
        Add a uniform degree logit model to the list of models.
        """
        self.models += [UniformModel(self.id, D=self.D, max_deg=self.max_deg)]
        self.model_short += 'u'

    def add_degree_model(self):
        """
        Add a degree logit model to the list of models.
        """
        self.models += [DegreeModel(self.id, D=self.D, max_deg=self.max_deg)]
        self.model_short += 'dd'

    def add_log_degree_model(self, bounds=None):
        """
        Add a log degree logit model to the list of models.
        """
        self.models += [LogDegreeModel(self.id, D=self.D, max_deg=self.max_deg, bounds=bounds)]
        self.model_short += 'ld'

    def add_poly_degree_model(self, k=2, bounds=None):
        """
        Add a poly degree logit model to the list of models.
        """
        self.models += [PolyDegreeModel(self.id, D=self.D, k=k, max_deg=self.max_deg, bounds=bounds)]
        self.model_short += 'pd%d' % k

    def add_uniform_fof_model(self):
        """
        Add a uniform fof logit model to the list of models.
        """
        self.models += [UniformFofModel(self.id, D=self.D, max_deg=self.max_deg)]
        self.model_short += 'uf'

    def add_log_degree_fof_model(self, bounds=None):
        """
        Add a log degree fof logit model to the list of models.
        """
        self.models += [LogDegreeFoFModel(self.id, D=self.D, max_deg=self.max_deg, bounds=bounds)]
        self.model_short += 'ldf'

    def add_log_fof_model(self, bounds=None):
        """
        Add a log degree logit model to the list of models.
        """
        self.models += [LogFofModel(self.id, D=self.D, max_deg=self.max_deg, bounds=bounds)]
        self.model_short += 'lf'

    def ll(self):
        """
        Compute log-likelihood for the mixture-model.
        LL = sum_i log ( sum_k p_ik * p_k)
        """
        ms = self.models  # shorthand
        K = len(ms)
        # initiate class probabilities if not already
        if len(self.pk) == 0:
            self.pk = {k: 1.0 / K for k in range(K)}
        probs = [0] * self.n
        for k in range(len(ms)):
            # compute sum of weighted probabilities for individual examples
            probs += ms[k].individual_likelihood(ms[k].u) * self.pk[k]
        # compute total log likelihood
        return -1 * np.sum(np.log(probs + util.log_smooth))

    def fit(self, n_rounds=20, etol=0.1, return_stats=False):
        """
        Fit the mixed model using a version of EM.

        Keyword arguments:

        n_rounds -- maximum number of iterations before the process stops
        etol -- minimum delta in total likelihood before the process stops
        return_stats -- return a pandas DataFrame with stats for every round
        """
        ms = self.models  # shorthand
        K = len(ms)  # number of classes
        T = []
        if K < 1:
            self.exception("not enough models specified for mixed model")
        # initate class probabilities
        self.pk = {k: 1.0 / K for k in range(K)}
        # store current total log-likelihood
        gamma = {k: [self.pk[k]] * self.n for k in range(K)}
        prev_ll = np.sum([ms[k].ll(w=gamma[k]) for k in range(K)])
        # run EM n_rounds times
        for i in range(n_rounds):
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
            # compute the total mixture's likelihood
            ll = self.ll()
            # gather stats for this round
            stats = [i]
            for k in range(K):
                stats += [self.pk[k], ms[k].u[0]]
            stats.append(ll)
            T.append(stats)
            # optionally print round info
            if self.vvv and i % 10 == 0:
                msg = "[%s/%3d]" % ("%3d", n_rounds)
                for k in range(1, K + 1):
                    msg += " (%s) pi_%d=%s u_%d=%s" % \
                           (ms[k-1].model_short, k, "%.3f", k, "%.2f")
                msg += " (*) tot_ll=%.4f"
                self.message(msg % tuple(stats))
            # compute current total likelihood
            delta = prev_ll - ll
            # stop if little difference
            if self.vvv and delta < etol:
                self.message("delta in ll (%.3f) < etol (%.3f), stopping" % (delta, etol))
                break
            # store new ll
            prev_ll = ll
        # print final results
        if self.vvv:
            self.message("u's  = [%s]" % ', '.join(['(%s:%.3f)' %
                         (ms[k].model_short, ms[k].u[0]) for k in range(K)]))
            self.message("pi's = [%s]" % ', '.join(['(%s:%.3f)' %
                         (ms[k].model_short, self.pk[k]) for k in range(K)]))
        # return the iteration stats
        if return_stats:
            # construct header
            header = ['i']
            for k in range(1, K + 1):
                header += ['p%d' % k, 'u%d' % k]
            header += ['tot_ll']
            return pd.DataFrame(T, columns=header)

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



class UniformModel(LogitModel):
    """
    This class represents a uniform logit model.
    There are no parameters.
    """
    def __init__(self, model_id, max_deg=50, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, max_deg=max_deg, bounds=((1, 1), ), D=D, vvv=vvv)
        self.model_type = 'uniform'
        self.model_short = 'u'

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the uniform logit model.
        Computes the likelihood for every data point (choice) separately.

        L(alpha, (x,C)) = 1 / |C|

        Contrary to the non-uniform models, we can actually compute the exact
        individual likelihood based on the total number of samples, as the
        individual likelihood for every unpicked choices is the same.
        """
        counts = self.D.groupby('choice_id')['y'].aggregate(len)
        return np.array(1.0 / counts)

    def grad(self, u=None, w=None):
        """
        Placeholder gradient function of the uniform fof logit model.
        Since there are no parameters, it always returns 0.
        """
        return np.array([0])


class DegreeModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    distinct coefficient beta_i for each individual degree i.
    """
    def __init__(self, model_id, max_deg=50, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, D=D, max_deg=max_deg, vvv=vvv)
        self.model_type = 'degree'
        self.model_short = 'dd'
        self.u = [1] * self.d  # current parameter values
        self.se = [None] * self.d  # current SE values

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
        Dt = Dt.groupby('deg')['prob'].aggregate(np.sum)
        # join with degree vector to take care of zeros
        Dd = pd.Series([0]*self.d, index=np.arange(self.d))
        return Dd.to_frame().join(Dt.to_frame()).prob.fillna(0)


class PolyDegreeModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    polynomial transformatin of degree: u[i] = sum_d ( i^d * theta[d] )
    """
    def __init__(self, model_id, max_deg=50, bounds=None, D=None, vvv=False, k=2):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, bounds=bounds, max_deg=max_deg, D=D, vvv=vvv)
        self.model_type = 'poly_degree'
        self.model_short = 'p%dd' % k
        self.u = [1] * k  # current parameter values
        self.se = [None] * k  # current SE values

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
        return np.array(np.exp(np.array(self.D.loc[self.D.y == 1, 'score']) - score_max - score_tot))

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


class LogDegreeModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    log transformation over degrees. The model has 1 parameter.
    """
    def __init__(self, model_id, max_deg=50, bounds=None, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, max_deg=max_deg, bounds=bounds, D=D, vvv=vvv)
        self.model_type = 'log_degree'
        self.model_short = 'ld'
        self.D['log_degree'] = np.log(self.D.deg + util.log_smooth)  # pre-log degree

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the log logit model.
        Computes the likelihood for every data point (choice) separately.

        L(alpha, (x,C)) = exp(alpha * log(k_x)) / sum_{y in C} exp(alpha * log(k_y))
        """
        # transform degree to score
        self.D['score'] = np.exp(u * np.log(self.D.deg + util.log_smooth))
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


class FeatureModel(LogitModel):
    """
    This class represents a multinomial logit model, with arbitrary features.
    The model has k parameters.
    """
    def __init__(self, model_id, bounds=None, D=None, vvv=False, features=['deg']):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, bounds=bounds, D=D, vvv=vvv)
        self.model_type = 'feature'
        self.model_short = 'f'
        self.features = features
        self.k = len(self.features)  # number of features
        self.u = [0] * len(self.features)


    def individual_likelihood(self, u):
        """
        Individual likelihood function of the log logit model.
        Computes the likelihood for every data point (choice) separately.

        L(u, (x,C)) = exp(u * X) / sum_{y in C} exp(u * X)

        u is actually theta, legacy name
        """
        # transform degree to score
        self.D['score'] = np.exp((self.D[self.features] * u).sum(axis=1))
        # compute total utility per case
        score_tot = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # compute probabilities of choices
        return np.array(self.D.loc[self.D.y == 1, 'score']) / np.array(score_tot)


    def grad(self, u=None, w=None):
        """
        Gradient function of log logit model.

        grad(theta_i, D) = sum_{(x,C) in D} [ x_i -
          (sum_{y in C} y_i*exp(theta*y)) /
          (sum_{y in C}     exp(theta*y))
        ]
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # transform degree to score
        self.D['score'] = np.exp((self.D[self.features] * u).sum(axis=1))
        # initialize empty gradient vector to append to
        grad = np.array([])
        # compute each k-specific gradient separately
        for f in self.features:
            # take log_degree for chosen examples
            choices = self.D.loc[self.D.y == 1, f]
            # compute 'numerator score'
            self.D['nscore'] = self.D['score'] * self.D[f]
            # compute numerator
            num = self.D.groupby('choice_id')['nscore'].aggregate(np.sum)
            # compute denominator
            denom = self.D.groupby('choice_id')['score'].aggregate(np.sum)
            # weight probabilities, add to grad matrix
            grad = np.append(grad, np.sum(w * (np.array(choices) - num/denom)))
        return -1 * grad



class UniformFofModel(LogitModel):
    """
    This class represents a uniform logit model with only friends of friends
    in the choice set. There are no parameters.
    """
    def __init__(self, model_id, max_deg=50, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, max_deg=max_deg, bounds=((1, 1), ), D=D, vvv=vvv)
        self.model_type = 'uniform_fof'
        self.model_short = 'uf'
        # pre-compute variables
        self.D['has'] = self.D.fof > 0  # has any FoF choices
        self.D['choose'] = 1 * (self.D['has'] & self.D.y == 1)  # chose an FoF node

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the uniform fof logit model.
        Computes the likelihood for every data point (choice) separately.

        L(alpha, (x,C)) = 1{x fof} / | #fof| if |#fof| > 0 else 1 / |C|

        Contrary to the non-uniform models, we can actually compute the exact
        individual likelihood based on the total number of samples, as the
        individual likelihood for every unpicked choices is the same.
        """
        # pre-group
        DFg = self.D.groupby('choice_id', as_index=False).agg(
            {'has': {'n': len, 'n_fof': np.sum}, 'choose': {'y': max}})
        return np.where(DFg.has.n_fof, DFg.choose.y / DFg.has.n_fof, 1.0 / DFg.has.n)

    def grad(self, u=None, w=None):
        """
        Placeholder gradient function of the uniform fof logit model.
        Since there are no parameters, it always returns 0.
        """
        return np.array([0])


class LogDegreeFoFModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    log transformation over degrees, but with only friends of friends
    in the choice set. The model has 1 parameter.
    """
    def __init__(self, model_id, max_deg=50, bounds=None, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, max_deg=max_deg, bounds=bounds, D=D, vvv=vvv)
        self.model_type = 'log_degree_fof'
        self.model_short = 'ldf'
        self.bounds = bounds  # bound the parameter
        # pre-compute variables
        self.D['has'] = self.D.fof > 0  # has any FoF choices
        self.D['choose'] = 1 * (self.D['has'] & self.D.y == 1)  # chose an FoF node
        self.D['log_degree'] = np.log(self.D.deg + util.log_smooth)  # pre-log degree
        DFg = self.D.groupby('choice_id', as_index=False).agg({'has': {'n_fof': np.sum}})
        self.elig = DFg.has.n_fof   # number of eligible FoFs

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the log logit model, for FoFs only.
        Computes the likelihood for every data point (choice) separately.
        This likelihood computation is quite involved, as it is a mix between
        the log degree model (in that it has alpha as a PA parameter),
        and the uniform fof model (in that it considers FoFs only).
        If there are not eligible FoFs, it considers all other nodes.
        """
        # 1) compute log degree scores for full choice set
        # transform degree to score
        self.D['score'] = np.exp(u * np.log(self.D.deg + util.log_smooth))
        # compute total utility per case
        score_tot = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # compute probabilities of choices
        scores_all = np.array(self.D.loc[self.D.y == 1, 'score']) / np.array(score_tot)
        # 2) compute log degree scores for FoFs only
        # set non-FoF scores to 0
        self.D['score'] = np.where(self.D['fof'] == 0, 0, self.D['score'])
        # compute total utility per case
        score_tot = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # compute probabilities of choices
        scores_fof = np.array(self.D.loc[self.D.y == 1, 'score']) / np.array(score_tot)
        # 3) actually construct the outcome vector, depending on the choice set
        return np.where(self.elig, scores_fof, scores_all)

    def grad(self, u=None, w=None):
        """
        Gradient function of log logit model, for FoFs only.
        Like the likelihood function, it mixes the gradients for 
        the log degree model (in that it has alpha as a PA parameter),
        and the uniform fof model (in that it considers FoFs only).
        If there are not eligible FoFs, it considers all other nodes.
        """
        # if no parameters specified, use the parameters of the object itself
        if u is None:
            u = self.u
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)

        # 1) construct the gradients as if it was a regular log-degree model
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
        grads_all = np.array(choices) - num/denom

        # 2) construct the gradients for FoF choices only
        # set non-FoF scores to 0
        self.D['score'] = np.where(self.D['fof'] == 0, 0, self.D['score'])
        # compute 'numerator score'
        self.D['nscore'] = self.D['score'] * self.D['log_degree']
        # compute numerator
        num = self.D.groupby('choice_id')['nscore'].aggregate(np.sum)
        # compute denominator
        denom = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        grads_fof = np.array(choices) - num/denom

        # actually construct the individual gradient vector, depending on the choice set,
        # and weight the samples
        return np.array([-1 * np.sum(w * np.where(self.elig, grads_fof, grads_all))])


class LogFofModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    log transformation over number of friends of friends.
    The model has 1 parameter.
    If there are no FoF's in the choice set, it looks like a uniform model
    TODO - the same constructor and data reading functions are used, which
           user a filter with max_deg. Perhaps should consider adding a
           parameter for max_fof. Maybe not necessary as we're not fitting
           a FofLogitModel
    """
    def __init__(self, model_id, max_deg=50, bounds=None, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, D=D, bounds=bounds, max_deg=max_deg, vvv=vvv)
        self.model_type = 'log_fof'
        self.model_short = 'lf'
        self.bounds = bounds  # bound the parameter
        self.D.loc[:,'log_fof'] = np.log(self.D.fof + util.log_smooth)  # pre-log fof

    def individual_likelihood(self, u):
        """
        Individual likelihood function of the log logit model.
        Computes the likelihood for every data point (choice) separately.

        L(alpha, (x,C)) = exp(alpha * log(k_x)) / sum_{y in C} exp(alpha * log(k_y))
        """
        # transform fof to score
        self.D['score'] = np.exp(u * self.D['log_fof'])
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
        # transform fof to score
        self.D['score'] = np.exp(u * self.D['log_fof'])
        # take log_fof for chosen examples
        choices = self.D.loc[self.D.y == 1, 'fof']
        # compute 'numerator score'
        self.D['nscore'] = self.D['score'] * self.D['fof']
        # compute numerator
        num = self.D.groupby('choice_id')['nscore'].aggregate(np.sum)
        # compute denominator
        denom = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # weight probabilities
        return np.array([-1 * np.sum(w * (np.array(choices) - num/denom))])
