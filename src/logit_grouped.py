import numpy as np
import pandas as pd
import util
from logit import LogitModel
from scipy.misc import logsumexp

"""

  This script contains model definitions and objective functions for
  multinomial logit modes on *grouped* data. The choice cases are
  represented as columns with the counts of nodes with degree i,
  and the choices are integers representing what the degree of the
  chosen node was.

"""


class DegreeLogitModelGrouped(LogitModel):
    """
    This class represents a multinomial logit model, with a
    distinct coefficient beta_i for each individual degree i.
    """
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, grouped=True, N=N, C=C, max_deg=max_deg, vvv=vvv)
        self.model_type = 'logit_degree_group'
        self.model_short = 'd'
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


class PolyLogitModelGrouped(LogitModel):
    """
    This class represents a multinomial logit model, with a
    polynomial functional form: [i] = sum_k ( i^k * theta[i] )
    """
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=False, bounds=None, k=2):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, grouped=True, N=N, C=C, max_deg=max_deg, vvv=vvv)
        self.model_type = 'logit_poly_group'
        self.model_short = 'p'
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
        score = np.array([util.poly_utilities(self.d, u)] * self.n)
        # combine log-sum-exp components
        return np.exp(score[range(self.n), self.C] - logsumexp(score, axis=1, b=self.N))

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
        score = np.array([util.poly_utilities(self.d, u)] * self.n)
        # make matrix of degrees to take power
        D = np.array([range(self.d)] * self.n)
        # initialize empty gradient vector to append to
        grad = np.array([])
        # compute gradient for every polynomial degree separately
        for k in range(len(u)):
            # compute 'numerator': power degree * group n * poly utility
            num = logsumexp(score, axis=1, b=np.power(D, k) * self.N)
            # compute 'denominator': group n * poly utility
            denom = logsumexp(score, axis=1, b=self.N)
            # score is degree choice ^ poly degree, normalized by division
            scores = self.C ** k - np.exp(num - denom)
            # sum over rows, potentially weighted
            grad = np.append(grad, np.sum(scores * w))
        return -1 * grad


class LogLogitModelGrouped(LogitModel):
    """
    This class represents a multinomial logit model, with a
    log transformation over degrees. The model has 1 parameter.
    """
    def __init__(self, model_id, N=None, C=None, max_deg=50, vvv=False, bounds=None):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, grouped=True, N=N, C=C, max_deg=max_deg, vvv=vvv)
        self.model_type = 'logit_log_group'
        self.model_short = 'l'
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
        score = np.exp(u * np.log(np.array([range(self.d)] * self.n) + util.log_smooth))
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
        D = np.log(np.array([range(self.d)] * self.n) + util.log_smooth)
        # compute numerator : log degree * group n * log utility
        num = np.sum(D * self.N * np.exp(u * D), axis=1)  # row sum
        # compute denominator : group n * log utility
        denom = np.sum(self.N * np.exp(u * D), axis=1)  # row sum
        # normalize by total summed up exponentiated utility
        scores = D[range(self.n), self.C] - num / denom
        # sum over rows, potentially weighted
        return -1 * np.array([np.sum(scores * w)])

