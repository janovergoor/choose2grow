import csv
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import util


"""

  This script contains the generic LogitModel and MixedLogit classes.
  They both have with data loading, model fitting, and output functionality.

"""

class LogitModel:
    """
    This class represents a generic logit model.
    """
    def __init__(self, model_id, grouped=True, max_deg=50, bounds=None,
                 N=None, C=None, D=None, vvv=0):
        """
        Constructor for a LogitModel object. The data can be provided directly,
        or it can be read in from file.

        Keyword arguments:

        model_id -- model_id can be either the file name from where the choices
            are read, or an idenfier for the current model
        grouped -- boolean signifiying whether the data is grouped (by degree),
            or if it contains individual rows (default: True)
        max_deg -- the max degree that will be considered (default: 50)
        N -- 2d-array representing choice sets, if the data is supplied
            directly in grouped format (default: None)
        C -- 1d-array representing chosen objects, if the data is supplied
            directly in grouped format (default: None)
        D -- 2d-array representing choice options, if the data is supplied
            directly in individual format (default: None)
        vvv -- int representing level of debug output [0:none, 1:some, 2:lots]

        If data is supplied directly, it can be done one of two formats:
        1) grouped: N is a n*d matrix representing the d-dimensional options
           for each of the n examples. C is an n*1 vector of choices.
        2) individual: D is a (n*i)x4 matrix, where each choice set has i
           choices, exactly one of which should be chosen. For every example
           we get the following covariates: [choice_id, Y, degree, n_fofs]
        """
        self.id = model_id
        self.vvv = vvv

        if grouped:
            # read data from file if the filename is not specified
            if N is not None and C is not None:
                if N.shape[0] != C.shape[0]:
                    self.exception("N and C do not fit together...")
                self.N = N
                self.C = C
            elif '.' in model_id:
                (self.N, self.C) = util.read_grouped_data(model_id, max_deg, vvv=vvv)
            else:
                self.exception("neither filename nor N and C are specified...")
            self.n = self.N.shape[0]  # number of examples
            self.d = self.N.shape[1]  # number of degrees considered (max_deg + 1)
        else:
            if D is not None:
                self.D = D
            elif '.' in model_id:
                self.D = util.read_individual_data(model_id, max_deg, vvv=vvv)
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


# ugly mid-file imports for MixedLogit class
from logit_grouped import *
from logit_individual import *

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
    def __init__(self, model_id, grouped=True, max_deg=50, N=None,
                 C=None, D=None, vvv=0):
        """
        Constructor for a MixedLogitModel object. It inherits from LogitModel,
        but is also composed of LogitModel modes. Like LogitModel, the data can
        be provided directly, or it can be read in from file.

        Keyword arguments:

        model_id -- model_id can be either the file name from where the choices
            are read, or an idenfier for the current model
        grouped -- boolean signifiying whether the data is grouped (by degree),
            or if it contains individual rows (default: True)
        max_deg -- the max degree that will be considered (default: 50)
        N -- 2d-array representing choice sets, if the data is supplied
            directly in grouped format (default: None)
        C -- 1d-array representing chosen objects, if the data is supplied
            directly in grouped format (default: None)
        D -- 2d-array representing choice options, if the data is supplied
            directly in individual format (default: None)
        vvv -- int representing level of debug output [0:none, 1:some, 2:lots]

        If data is supplied directly, it can be done one of two formats:
        1) grouped: N is a n*d matrix representing the d-dimensional options
           for each of the n examples. C is an n*1 vector of choices.
        2) individual: D is a (n*i)x4 matrix, where each choice set has i
           choices, exactly one of which should be chosen. For every example
           we get the following covariates: [choice_id, Y, degree, n_fofs]
        """
        LogitModel.__init__(self, model_id, grouped=grouped, max_deg=max_deg,
                            N=N, C=C, D=D, vvv=vvv)
        self.grouped = grouped
        self.model_type = 'mixed_logit'
        self.max_deg = max_deg
        self.vvv = vvv
        self.models = []  # list of models
        self.pk = {}  # class probabilities
        self.model_short = ''

    def add_uniform_degree_model(self):
        """
        Add a uniform degree logit model to the list of models.
        """
        if self.grouped:
            self.exception("UniformDegreeModel is not implemented for grouped data!")
        else:
            self.models += [UniformDegreeModel(self.id, D=self.D, max_deg=self.max_deg)]
        self.model_short += 'ud'

    def add_degree_model(self):
        """
        Add a degree logit model to the list of models.
        """
        if self.grouped:
            self.models += [DegreeLogitModelGrouped(self.id, N=self.N, C=self.C,
                                                    max_deg=self.max_deg)]
        else:
            self.models += [DegreeModel(self.id, D=self.D, max_deg=self.max_deg)]
        self.model_short += 'dd'

    def add_log_degree_model(self, bounds=None):
        """
        Add a log degree logit model to the list of models.
        """
        if self.grouped:
            self.models += [LogDegreeModelGrouped(self.id, N=self.N, C=self.C,
                                max_deg=self.max_deg, bounds=bounds)]
        else:
            self.models += [LogDegreeModel(self.id, D=self.D,
                                           max_deg=self.max_deg, bounds=bounds)]
        self.model_short += 'ld'

    def add_poly_degree_model(self, k=2, bounds=None):
        """
        Add a poly degree logit model to the list of models.
        """
        if self.grouped:
            self.models += [PolyDegreeModelGrouped(self.id, N=self.N, C=self.C, k=k,
                                max_deg=self.max_deg, bounds=bounds)]
        else:
            self.models += [PolyDegreeModel(self.id, D=self.D, k=k,
                                            max_deg=self.max_deg, bounds=bounds)]
        self.model_short += 'pd%d' % k

    def add_uniform_fof_model(self):
        """
        Add a uniform fof logit model to the list of models.
        """
        if self.grouped:
            self.exception("UniformFofModel is not implemented for grouped data!")
        else:
            self.models += [UniformFofModel(self.id, D=self.D, max_deg=self.max_deg)]
        self.model_short += 'uf'

    def add_log_fof_model(self, bounds=None):
        """
        Add a log degree logit model to the list of models.
        """
        if self.grouped:
            self.exception("LogFofModel is not implemented for grouped data!")
        else:
            self.models += [LogFofModel(self.id, D=self.D,
                                        max_deg=self.max_deg, bounds=bounds)]
        self.model_short += 'lf'

    def ll(self, gamma):
        """
        Compute log-likelihood for the mixture-model.
        LL = sum_i log ( sum_k p_ik * gamma_ik)

        Keyword arguments:

        gamma -- dictionary of gamma vectors for each class k
        """
        ms = self.models  # shorthand
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
            ll = self.ll(gamma)
            # gather stats for this round
            stats = [i]
            for k in range(K):
                stats += [self.pk[k], ms[k].u[0], ms[k].ll(w=gamma[k])]
            stats.append(ll)
            T.append(stats)
            # optionally print round info
            if self.vvv and i % 10 == 0:
                msg = "[%s/%3d]" % ("%3d", n_rounds)
                for k in range(1, K + 1):
                    msg += " (%s) pi_%d=%s u_%d=%s ll_%d=%s" % \
                           (ms[k-1].model_short, k, "%.3f", k, "%.2f", k, "%.2f")
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
                header += ['p%d' % k, 'u%d' % k, 'll%d' % k]
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
