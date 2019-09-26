import csv
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from copy import deepcopy
import util


"""

  This script contains the generic LogitModel and MixedLogit classes.
  These both have data loading, model fitting, and output functionality.
  It includes model definitions and objective functions for individual modes.
  The examples (data) are represented as (choice_id, Y, degree, n_fofs).

"""

class LogitModel:
    """
    This class represents a generic logit model.
    """
    def __init__(self, model_id, bounds=None, choice_set=None, D=None, vvv=0, sw=None):
        """
        Constructor for a LogitModel object. The data can be provided directly,
        or it can be read in from file.

        Keyword arguments:

        model_id -- model_id can be either the file name from where the choices
            are read, or an idenfier for the current model
        bounds -- tuple of tuples for every feature with bounds for the estimates
        choice_set -- name of column containing the reduced choice set (if any)
        D -- 2d-array representing choice options
        vvv -- int representing level of debug output [0:none, 1:some, 2:lots]
        sw -- name of the column containing the sample weights

        If data is supplied directly, D is a (n*i)x4 matrix, where each
        choice set has i choices, exactly one of which should be chosen.
        """
        self.id = model_id
        self.vvv = vvv

        if D is not None:
            self.D = D
        elif '.' in model_id:
            self.D = util.read_data(model_id, vvv=vvv)
        else:
            self.exception("neither filename nor D are specified..")
        if choice_set is not None:
            # check whether choice_id column exists
            if choice_set not in D.columns:
                self.exception("choice_set column (%s) do not exist in the data.." % choice_set)
            else:
                self.D['choice_set'] = self.D[choice_set]
            # check whether choice_id column contains values outside of [0,1]
            if not np.array_equal(np.sort(self.D['choice_set'].unique()), [0, 1]):
                self.exception("choice_set column (%s) contains value outside of [0, 1].." % choice_set)
        else:
            self.D['choice_set'] = 1
        self.n = len(set(self.D.choice_id))  # number of examples
        if sw is not None:
            if sw in list(D):
                self.D['sw'] = D[sw]
            else:
                self.exception("sample weight column %s doesn't exist in the data.." % sw)
        else:
                self.D['sw'] = np.array([1] * self.D.shape[0])

        # initiate the rest of the parameters
        self.n_it = 0  # number of iterations for optimization
        self.theta = [1]  # current parameter value
        self.se = [None]  # current SE value
        self.bounds = bounds  # whether there are bounds for the parameters

    def ll(self, theta=None, w=None):
        """
        Generic log-likelihood function. It computes individual likelihood
        scores for each example using a model-specifc formula, and computes a
        (weighted) sum of their logs.
        """
        # if no parameters specified, use the parameters of the object itself
        if theta is None:
            theta = self.theta
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # compute individual likelihood scores
        scores = self.individual_likelihood(theta)
        # set individual likelihood to 0 if choice was outside of choice set
        scores = scores * self.D.query('y == 1')['choice_set']
        # add smoothing for u=0 choices
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
            res = minimize(lambda x: self.ll(x, w=w), self.theta,
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
            res = minimize(lambda x: self.ll(x, w=w), self.theta,
                           method='L-BFGS-B', callback=print_iter,
                           bounds=self.bounds)
        # store the resulting parameters
        self.theta = res.x
        if self.vvv > 0:
            self.message("parameters after fitting: " + str(self.theta))

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
            for i in range(len(self.theta)):
                row = [self.id,
                       self.model_type,
                       1,
                       i,
                       self.theta[i],
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
    def __init__(self, model_id, D=None, vvv=0):
        """
        Constructor for a MixedLogitModel object. It inherits from LogitModel,
        but is also composed of LogitModel modes. Like LogitModel, the data can
        be provided directly, or it can be read in from file.

        Keyword arguments:

        model_id -- model_id can be either the file name from where the choices
            are read, or an idenfier for the current model
        D -- 2d-array representing choice options
        vvv -- int representing level of debug output [0:none, 1:some, 2:lots]
        """
        LogitModel.__init__(self, model_id, D=D, vvv=vvv)
        self.model_type = 'mixed_logit'
        self.vvv = vvv
        self.models = []  # list of models
        self.pk = {}  # class probabilities
        self.model_short = ''


    def add_degree_model(self, max_deg=50, choice_set=None):
        """
        Add a degree logit model to the list of models.
        """
        self.models += [DegreeModel(self.id, D=deepcopy(self.D), max_deg=max_deg,
                                    choice_set=choice_set)]
        self.model_short += 'dd'

    def add_log_degree_model(self, bounds=None, choice_set=None):
        """
        Add a log degree logit model to the list of models.
        """
        self.models += [LogDegreeModel(self.id, D=deepcopy(self.D), bounds=bounds,
                                       choice_set=choice_set)]
        self.model_short += 'ld'

    def add_feature_model(self, features=[''], bounds=None, choice_set=None, sw=None):
        """
        Add a generic feature logit model to the list of models.
        """
        self.models += [FeatureModel(self.id, D=deepcopy(self.D), features=features,
                                     bounds=bounds, choice_set=choice_set, sw=sw)]
        self.model_short += 'f'


    def add_uniform_fof_model(self):
        """
        Add a uniform fof logit model to the list of models.
        """
        self.models += [UniformFofModel(self.id, D=self.D)]
        self.model_short += 'uf'

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
            probs += ms[k].individual_likelihood(ms[k].theta) * self.pk[k]
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
            probs = {k: ms[k].individual_likelihood(ms[k].theta) for k in range(K)}
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
                stats += [self.pk[k], ms[k].theta[0]]
            stats.append(ll)
            T.append(stats)
            # optionally print round info
            if self.vvv and i % 1 == 0:
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
            self.message("params's  = [%s]" % ', '.join(['(%s:%.3f)' %
                         (ms[k].model_short, ms[k].theta[0]) for k in range(K)]))
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
                for i in range(len(m.theta)):
                    row = [fn,
                           m.model_type,
                           "%.3f" % self.pk[k],
                           i,
                           m.theta[i],
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


class DegreeModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    distinct coefficient beta_i for each individual degree i.

    max_deg -- the max degree that will be considered (default: 50)
    """
    def __init__(self, model_id, max_deg=50, choice_set=None, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, D=D, vvv=vvv, choice_set=choice_set)
        self.max_deg = max_deg + 1  # number of degrees considered
        self.model_type = 'degree'
        self.model_short = 'dd'
        self.theta = [1] * self.max_deg  # current parameter values
        self.se = [None] * self.max_deg  # current SE values

    def individual_likelihood(self, theta):
        """
        Individual likelihood function of the degree logit model.
        Computes the likelihood for every data point (choice) separately.

        L(theta, (x,C)) = exp(theta_{k_x}) / sum_{y in C} exp(theta_{k_y})
        """
        # assign exponentiated utilities to all cases
        self.D['score'] = np.exp(theta)[self.D.deg]
        # compute total utility per case
        score_tot = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # compute probabilities of choices
        return np.array(self.D.loc[self.D.y == 1, 'score']) / np.array(score_tot)

    def grad(self, theta=None, w=None):
        """
        Gradient function of the degree logit model.

        grad_d(theta, D) = sum_{(x,C) in D} [ 1{k_x = d} -
          (sum_{y in C} 1{k_y = d}*exp(theta_k_y)) /
          (sum_{y in C}            exp(theta_k_y))
        ]
        """
        # if no parameters specified, use the parameters of the object itself
        if theta is None:
            theta = self.theta
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # assign weights to choice sets
        W = pd.DataFrame(data={'choice_id': self.D.choice_id.unique(), 'w': w})
        # assign utilities to all cases
        self.D['score'] = np.exp(theta)[self.D.deg]
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
        Dd = pd.Series([0] * self.max_deg, index=np.arange(self.max_deg))
        return Dd.to_frame().join(Dt.to_frame()).prob.fillna(0)


class LogDegreeModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    log transformation over degrees. The model has 1 parameter.
    """
    def __init__(self, model_id, bounds=None, choice_set=None, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, bounds=bounds, D=D, vvv=vvv, choice_set=choice_set)
        self.model_type = 'log_degree'
        self.model_short = 'ld'
        self.D['log_degree'] = np.log(self.D.deg + util.log_smooth)  # pre-log degree

    def individual_likelihood(self, theta):
        """
        Individual likelihood function of the log logit model.
        Computes the likelihood for every data point (choice) separately.

        L(theta, (x,C)) = exp(theta * log(k_x)) / sum_{y in C} exp(theta * log(k_y))
        """
        # transform degree to score
        self.D['score'] = np.exp(theta * np.log(self.D.deg + util.log_smooth))
        # compute total utility per case
        score_tot = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # compute probabilities of choices
        return np.array(self.D.loc[self.D.y == 1, 'score']) / np.array(score_tot)

    def grad(self, theta=None, w=None):
        """
        Gradient function of log logit model.

        grad(theta, D) = sum_{(x,C) in D} [ theta*ln(k_x) -
          (sum_{y in C} ln(k_y)*exp(theta*ln(k_y))) /
          (sum_{y in C}         exp(theta*ln(k_y)))
        ]
        """
        # if no parameters specified, use the parameters of the object itself
        if theta is None:
            theta = self.theta
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # transform degree to score
        self.D['score'] = np.exp(theta * self.D['log_degree'])
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
    def __init__(self, model_id, bounds=None, choice_set=None, D=None, vvv=False, sw=None, features=['deg']):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, bounds=bounds, D=D, vvv=vvv, sw=sw, choice_set=choice_set)
        if not set(features).issubset(D.columns):
            self.exception("some features (%s) do not exist in the data.." % ','.join(features))
        self.model_type = 'feature'
        self.model_short = 'f'
        self.features = features
        self.k = len(self.features)  # number of features
        self.theta = np.random.normal(0, 0.1, size=self.k)  # randomly initate parameter values
        self.se = [None] * self.k


    def individual_likelihood(self, theta):
        """
        Individual likelihood function of the log logit model.
        Computes the likelihood for every data point (choice) separately.

        L(theta, (x,C)) = exp(theta * X) / sum_{y in C} exp(theta * X)
        """
        # compute score
        self.D['score'] = np.exp((self.D[self.features] * theta).sum(axis=1) - np.log(self.D['sw']))
        # exclude items outisde the choice set
        self.D['score'] = self.D['score'] * self.D['choice_set']
        # compute total utility per case
        score_tot = self.D.groupby('choice_id')['score'].aggregate(np.sum)
        # compute probabilities of choices
        return np.array(self.D.loc[self.D.y == 1, 'score']) / np.array(score_tot)


    def grad(self, theta=None, w=None):
        """
        Gradient function of log logit model.

        grad(theta_i, D) = sum_{(x,C) in D} [ x_i -
          (sum_{y in C} y_i*exp(theta*y)) /
          (sum_{y in C}     exp(theta*y))
        ]
        """
        # if no parameters specified, use the parameters of the object itself
        if theta is None:
            theta = self.theta
        # if no weights specified, default to 1
        if w is None:
            w = np.array([1] * self.n)
        # compute score
        self.D['score'] = np.exp((self.D[self.features] * theta).sum(axis=1) - np.log(self.D['sw']))
        # exclude items outisde the choice set
        self.D['score'] = self.D['score'] * self.D['choice_set']
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
            grad = np.append(grad, np.sum(w * (np.array(choices) - num / denom)))
        return -1 * grad


class UniformFofModel(LogitModel):
    """
    This class represents a uniform logit model with only friends of friends
    in the choice set. There are no parameters.
    """
    def __init__(self, model_id, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, bounds=((1, 1), ), D=D, vvv=vvv)
        self.model_type = 'uniform_fof'
        self.model_short = 'uf'
        # pre-compute variables
        self.D['has'] = self.D.fof > 0  # has any FoF choices
        self.D['choose'] = 1 * (self.D['has'] & self.D.y == 1)  # chose an FoF node

    def individual_likelihood(self, theta):
        """
        Individual likelihood function of the uniform fof logit model.
        Computes the likelihood for every data point (choice) separately.

        L(theta, (x,C)) = 1{x fof} / | #fof| if |#fof| > 0 else 1 / |C|

        Contrary to the non-uniform models, we can actually compute the exact
        individual likelihood based on the total number of samples, as the
        individual likelihood for every unpicked choices is the same.
        """
        # pre-group
        DFg = self.D.groupby('choice_id', as_index=False).agg(
            {'has': {'n': len, 'n_fof': np.sum}, 'choose': {'y': max}})
        return np.where(DFg.has.n_fof, DFg.choose.y / DFg.has.n_fof, 1.0 / DFg.has.n)

    def grad(self, theta=None, w=None):
        """
        Placeholder gradient function of the uniform fof logit model.
        Since there are no parameters, it always returns 0.
        """
        return np.array([0])
