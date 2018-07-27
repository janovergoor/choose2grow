import numpy as np
import pandas as pd
import util
from logit import LogitModel

"""

  This script contains model definitions and objective functions for
  multinomial logit models on *individual* data. The examples are
  represented as (choice_id, Y, degree, n_fofs).

  TODO - incorporate / implement FOF modes
"""


class DegreeLogitModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    distinct coefficient beta_i for each individual degree i.
    """
    def __init__(self, model_id, D=None, max_deg=50, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, grouped=False, D=D, max_deg=max_deg, vvv=vvv)
        self.model_type = 'logit_degree'
        self.model_short = 'd'
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


class PolyLogitModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    polynomial functional form: u[i] = sum_d ( i^d * theta[d] )
    """
    def __init__(self, model_id, D=None, max_deg=50, vvv=False, k=2, bounds=None):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, grouped=False, D=D, max_deg=max_deg, vvv=vvv)
        self.model_type = 'logit_poly'
        self.model_short = 'p'
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


class LogLogitModel(LogitModel):
    """
    This class represents a multinomial logit model, with a
    log transformation over degrees. The model has 1 parameter.
    """
    def __init__(self, model_id, D=None, max_deg=50, vvv=False, bounds=None):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, grouped=False, D=D, max_deg=max_deg, vvv=vvv)
        self.model_type = 'logit_log'
        self.model_short = 'l'
        self.u = [1]  # current parameter value
        self.se = [None]  # current SE value
        self.bounds = bounds  # bound the parameter
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

