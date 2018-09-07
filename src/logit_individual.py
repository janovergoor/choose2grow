import numpy as np
import pandas as pd
import util
from logit import LogitModel

"""

  This script contains model definitions and objective functions for
  multinomial logit models on *individual* data. The examples are
  represented as (choice_id, Y, degree, n_fofs).

"""


class UniformModel(LogitModel):
    """
    This class represents a uniform logit model.
    There are no parameters.
    """
    def __init__(self, model_id, max_deg=50, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, grouped=False, max_deg=max_deg, bounds=((1, 1), ), D=D, vvv=vvv)
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
        # if we read the C series before, use that
        if getattr(self, 'C', None) is not None:
            counts = self.C.n_elig
        # otherwise, use the number of samples
        else:
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
        LogitModel.__init__(self, model_id, grouped=False, D=D, max_deg=max_deg, vvv=vvv)
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

        TODO - implement C-weighting negative samples
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
        LogitModel.__init__(self, model_id, grouped=False, bounds=bounds, max_deg=max_deg, D=D, vvv=vvv)
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
        LogitModel.__init__(self, model_id, grouped=False, max_deg=max_deg, bounds=bounds, D=D, vvv=vvv)
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


class UniformFofModel(LogitModel):
    """
    This class represents a uniform logit model with only friends of friends
    in the choice set. There are no parameters.
    """
    def __init__(self, model_id, max_deg=50, D=None, vvv=False):
        """
        Constructor inherits from LogitModel.
        """
        LogitModel.__init__(self, model_id, grouped=False, max_deg=max_deg, bounds=((1, 1), ), D=D, vvv=vvv)
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
        # if we read the C series before, use that
        if getattr(self, 'C', None) is not None:
            # pre-group
            DFg = self.D.groupby('choice_id', as_index=False).agg(
                {'choose': {'y': max}, 'n_elig': {'m': max}, 'n_elig_fof': {'m': max}})
            return np.where(DFg.n_elig_fof.m, DFg.choose.y / DFg.n_elig_fof.m, 1.0 / DFg.n_elig.m)
        # otherwise, use the number of samples
        else:
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
        LogitModel.__init__(self, model_id, grouped=False, max_deg=max_deg, bounds=bounds, D=D, vvv=vvv)
        self.model_type = 'log_degree_fof'
        self.model_short = 'ldf'
        self.bounds = bounds  # bound the parameter
        # pre-compute variables
        self.D['has'] = self.D.fof > 0  # has any FoF choices
        self.D['choose'] = 1 * (self.D['has'] & self.D.y == 1)  # chose an FoF node
        self.D['log_degree'] = np.log(self.D.deg + util.log_smooth)  # pre-log degree
        # if we read the C series before, use that
        if getattr(self, 'C', None) is not None:
            DFg = self.D.groupby('choice_id', as_index=False).agg({'n_elig_fof': {'m': max}})
            self.elig = DFg.n_elig_fof.m  # number of eligible FoFs
        # otherwise, use the number of samples
        else:
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
        LogitModel.__init__(self, model_id, grouped=False, D=D, bounds=bounds, max_deg=max_deg, vvv=vvv)
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
        if getattr(self, 'C', None) is not None:
            self.D.loc[self.D.y != 1, 'score'] *= self.D.loc[self.D.y != 1, 'C']
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
