import random
import os
import torch
import time
import numpy as np
from collections import Counter
from itertools import chain, repeat


class DirectedMultiGraph(object):
  '''
  Directed graph instance with fixed number of nodes that preserves the edge
  formations in chronological order with adjacency stored
  both as two lists of Counter (to and fron) and as edges list.
  '''

  def __init__(self, num_nodes):
    self.num_nodes = num_nodes
    self.adjacency_to = [Counter() for _ in range(num_nodes)]
    self.adjacency_from = [Counter() for _ in range(num_nodes)]
    self.in_degs = np.zeros(num_nodes)
    self.out_degs = np.zeros(num_nodes)
    self.edges_list = []

  def add_edge(self, actor, target):
    self.out_degs[actor] += 1
    self.in_degs[target] += 1
    self.adjacency_to[actor][target] += 1
    self.adjacency_from[target][actor] += 1
    self.edges_list.append((actor, target))

  def grow_erdos_renyi(self, num_edges):
    '''
    Grow an Erdos-Renyi G(n,m) graph, where n is fixed to self.num_nodes and m
    is num_edges.
    '''
    actors = np.random.randint(0, self.num_nodes, size=num_edges)
    targets = (np.random.randint(1, self.num_nodes, size=num_edges) + actors) % self.num_nodes
    for actor, target in zip(actors, targets):
      self.add_edge(actor, target)

  def neg_samp_by_locality(self, actor, target, num_neg=24, max_num_local_sample=[8,8]):
    hop1 = {n for n in chain(self.adjacency_to[actor], self.adjacency_from[actor])}
    hop2 = {nn for n in chain(self.adjacency_to[actor], self.adjacency_from[actor])\
               for nn in chain(self.adjacency_to[n], self.adjacency_from[n])}      
    for n in hop1:
      hop2.discard(n)
    hop1.discard(actor)
    hop2.discard(actor)
    result, lnsw = [target], []
    n1, s1 = len(hop1), min(max_num_local_sample[0], len(hop1))
    n2, s2 = len(hop2), min(max_num_local_sample[1], len(hop2))
    n3, s3 = self.num_nodes - 1 - s1 - s2, num_neg + 1 - s1 - s2
    target_group = 3
    if target in hop1:
      lnsw.append(np.log(n1/s1))
      hop1.remove(target)
      target_group = 1
    elif target in hop2:
      lnsw.append(np.log(n2/s2))
      hop2.remove(target)
      target_group = 2
    else:
      lnsw.append(np.log(n3/s3))
    if hop1:
      result.extend(np.random.choice(list(hop1), replace=False, size=s1-int(target_group==1)))
      lnsw.extend(repeat(np.log(n1/s1), s1-int(target_group==1)))
    if hop2:
      result.extend(np.random.choice(list(hop2), replace=False, size=s2-int(target_group==2)))
      lnsw.extend(repeat(np.log(n2/s2), s2-int(target_group==2)))
    if s3 > int(target_group==3):
      result.extend(randint_excluding(0, self.num_nodes, chain(hop1, hop2, [actor, target]),\
                                      size=s3-int(target_group==3)))
      lnsw.extend(repeat(np.log(n3/s3), s3-int(target_group==3)))

    return result, lnsw

  def extract_feature(self, actor, candidates):
    num_i_to_to_j_ctr = Counter()
    is_ffof_j_ctr = Counter()

    for n in self.adjacency_to[actor]:
      for nn in self.adjacency_to[n]:
        num_i_to_to_j_ctr[nn] += 1

    for n in chain(self.adjacency_to[actor], self.adjacency_from[actor]):
      is_ffof_j_ctr[n] = 1
      for nn in chain(self.adjacency_to[n], self.adjacency_from[n]):
        is_ffof_j_ctr[nn] = 1

    in_degs = np.array([self.in_degs[c] for c in candidates])
    num_i_to_j = np.array([self.adjacency_to[actor][c] for c in candidates])
    num_i_from_j = np.array([self.adjacency_from[actor][c] for c in candidates])
    num_i_to_to_j = np.array([num_i_to_to_j_ctr[c] for c in candidates])
    is_ffof_j = np.array([is_ffof_j_ctr[c] for c in candidates])

    log_in_degrees = np.log(in_degs + (in_degs < 0.5).astype(int))
    log_i_to_j = np.log(num_i_to_j + (num_i_to_j < 0.5).astype(int))
    log_i_from_j = np.log(num_i_from_j + (num_i_from_j < 0.5).astype(int))
    log_i_to_to_j = np.log(num_i_to_to_j + (num_i_to_to_j < 0.5).astype(int))

    return np.array([log_in_degrees, log_i_to_j, log_i_from_j, log_i_to_to_j,\
                     (in_degs > 0.5).astype(int),      (num_i_to_j > 0.5).astype(int),\
                     (num_i_from_j > 0.5).astype(int), (num_i_to_to_j > 0.5).astype(int),\
                     is_ffof_j.astype(int)])

  def extract_feature_all_nodes(self, actor):
    num_i_to_j = np.zeros(self.num_nodes)
    num_i_from_j = np.zeros(self.num_nodes)
    num_i_to_to_j = np.zeros(self.num_nodes)
    is_ffof_j = np.zeros(self.num_nodes)

    for n in self.adjacency_to[actor]:
      num_i_to_j[n] = self.adjacency_to[actor][n]
    for n in self.adjacency_from[actor]:
      num_i_from_j[n] = self.adjacency_from[actor][n]

    for n in self.adjacency_to[actor]:
      for nn in self.adjacency_to[n]:
        num_i_to_to_j[nn] += 1

    for n in chain(self.adjacency_to[actor], self.adjacency_from[actor]):
      is_ffof_j[n] = 1
      for nn in chain(self.adjacency_to[n], self.adjacency_from[n]):
        is_ffof_j[nn] = 1

    log_in_degrees = np.log(self.in_degs + (self.in_degs < 0.5).astype(int))
    log_i_to_j = np.log(num_i_to_j + (num_i_to_j < 0.5).astype(int))
    log_i_from_j = np.log(num_i_from_j + (num_i_from_j < 0.5).astype(int))
    log_i_to_to_j = np.log(num_i_to_to_j + (num_i_to_to_j < 0.5).astype(int))

    return np.array([log_in_degrees, log_i_to_j, log_i_from_j, log_i_to_to_j,\
                    (self.in_degs > 0.5).astype(int), (num_i_to_j > 0.5).astype(int),\
                    (num_i_from_j > 0.5).astype(int), (num_i_to_to_j > 0.5).astype(int),\
                    is_ffof_j.astype(int)]) # <---- used as a flag for generating mixed model


class MNLogit(object):

  def __init__(self, num_threads=48):
    torch.set_num_threads(num_threads)

  def data(self, Xs, ys, sws=None):
    '''
    Ingest data into the model

    Dataset:
    - Xs: An np.ndarray of shape (N * C * D) -- The feature tensor
      N entries, C candidates in each entry, D features for each candidate.
    - ys: An np.ndarray of shape (N) -- The vector of labels
      One label for each entry. The label is the index of the selected candidate
    - ws: An np.ndarray of shape (N * C) -- The log stratified sampling weight
      N entries, C candidates in each entry, 1 weight for each candidate.
      Can be None
    '''

    self.data_len = Xs.shape[0]
    self.num_classes = Xs.shape[1]
    self.num_features = Xs.shape[2]
    ys_oh = np.zeros((self.data_len, self.num_classes))
    ys_oh[np.arange(self.data_len), ys] += 1
    self.X = torch.Tensor(Xs).to(dtype=torch.double)
    self.y = torch.Tensor(ys).to(dtype=torch.long)
    self.y_oh = torch.Tensor(ys_oh).to(dtype=torch.double)
    self.sw = torch.Tensor(sws).to(dtype=torch.double)
    self.w = torch.Tensor(np.zeros(self.num_features)).to(dtype=torch.double)
    self.B = 1e-3 * torch.Tensor(np.eye(self.num_features)).to(dtype=torch.double)
    self.num_iter = 0

  def fit(self, max_num_iter=30, clip=1, clip_norm_ord=2, itol=1e-7, reg=0, reg_ord=1, verbose_function=None):
    '''
    Fit multinomial logit choice model with the dataset

    Optimize parameters:
    - max_num_iter: The maximal number of iteration
    - itol: threshold -- stop when gradient norm is smaller than thresh
    - clip: clipping the step size so it has the norm <= clip
    - clip_norm_ord: the order of the norm used for clipping step size
    - verbose: Print the process
    '''
    t0 = time.time()

    for i in range(max_num_iter):
      t1 = time.time()
      info = self._step(clip, clip_norm_ord, reg, reg_ord)
      info['num_iter'] = i+1
      info['iter_time'] = time.time() - t1
      info['total_time'] = time.time() - t0

      if verbose_function is not None:
        verbose_function(self, info)

      self.num_iter += 1

      if info['inc_norm'] < itol or np.linalg.norm(self.w, ord=np.inf) > 25:
        return

  def eval(self, Xs, ys):
    '''
    Evaluate the dataset by classification accuracy
    '''

    result = (ys == np.argmax(Xs.dot(self.w.numpy()).reshape(-1, self.num_classes), axis=1))
    print("Accuracy: {} / {} ({:.4f})".format(np.sum(result), len(ys), np.sum(result)/len(ys)))

  def _loss(self, w):
    score = torch.nn.LogSoftmax(dim=1)(torch.matmul(self.X, w) + self.sw)
    return -torch.sum(score[torch.arange(self.data_len, dtype=torch.long), self.y])

  def _grad(self, w):
    score = torch.nn.LogSoftmax(dim=1)(torch.matmul(self.X, w) + self.sw)
    dscore = - self.y_oh + torch.exp(score)
    return torch.mm(dscore.view(1,-1), self.X.view(-1, self.num_features)).view(-1) ## 1 x N*C dot N*C x D


  def _line_search(self, p, loss, dw, max_iter=40):
    c1, c2 = 0.001, 0.9
    a = 0
    t = 1
    t0 = np.linalg.norm(p.numpy(), ord=2) / np.sqrt(self.num_features)
    b = None
    for i in range(max_iter):
      if self._loss(self.w + (t/t0) * p).numpy() > (loss + c1 * (t/t0) * p.dot(dw)).numpy():
        b = t
        t = (a + b)/2
        print(t)
      elif p.dot(self._grad(self.w + (t/t0) * p)).numpy() < c2 * p.dot(dw).numpy():
        a = t
        t = 2 * a if b is None else (a + b)/2
        print(t)
      else:
         return t
    return t

  def _step(self, clip=1, clip_norm_ord=2, reg=0, reg_ord=2):
    score = torch.nn.LogSoftmax(dim=1)(torch.matmul(self.X, self.w) + self.sw)
    loss = -torch.sum(score[torch.arange(self.data_len, dtype=torch.long), self.y])
    dscore = - self.y_oh + torch.exp(score)
    if reg_ord == 2:
      dw = torch.mm(dscore.view(1,-1), self.X.view(-1, self.num_features)).view(-1) + 2 * self.data_len * reg * self.w ## 1 x N*C dot N*C x D
    elif reg_ord == 1:
      dw = torch.mm(dscore.view(1,-1), self.X.view(-1, self.num_features)).view(-1) + self.data_len * reg * torch.sign(self.w) ## 1 x N*C dot N*C x D
    p = -torch.mv(self.B, dw)
    pnorm = torch.norm(p, p=clip_norm_ord)
    if pnorm.numpy() > clip:
      # t = self._line_search(p, loss, dw)
      s = p * clip / pnorm
    else:
      s = p
    self.w += s
    y = self._grad(self.w + s) - dw
    sy = s.dot(y)
    self.B += ((sy + y.dot(torch.mv(self.B,y))) / (sy * sy)) * torch.ger(s,s) - \
              (torch.ger(torch.mv(self.B,y),s) + torch.ger(s, torch.mv(torch.t(self.B),y))) / sy

    return {'total_loss':loss.numpy(),
            'avg_loss':loss.numpy()/self.data_len,
            'avg_grad_norm':torch.norm(dw).numpy()/self.data_len,
            'inc_norm':pnorm.numpy(),
            'se':np.sqrt(np.diag(self.B.numpy())) if pnorm.numpy() < clip else None}

  def get_model_info(self):
    return {'weights':self.w.numpy(),
            'se':np.sqrt(np.diag(self.B.numpy())),
            'avg_loss':self._loss(self.w).numpy() / self.data_len,
            'num_iter':self.num_iter}


def randint_excluding(low, high, excluding, size=1):
    if size == 0:
        return []
    exclusion = {num for num in excluding if low <= num < high}
    if size > high - low - len(exclusion):
        raise RuntimeError("Requested sample size larger than population.")
    if size == high - low - len(exclusion):
        return [x for x in range(low, high) if x not in exclusion]
    result = []
    while len(result) < size:
        rand_size = 1 + int(((size - len(result)) * (high-low) / (high-low-len(exclusion)+1)) // 1)
        buf = random.sample(range(low, high), min(rand_size, high - low))
        for num in buf:
            if num not in exclusion:
                result.append(num)
                if len(result) == size:
                    return result
                exclusion.add(num)
    return result


def write_info(PATH, info):
    with open(os.path.join(PATH, 'info'), 'w') as f:
        for k, v in info.items():
            f.write("{}\t{}\n".format(k, v))


def parse_info(PATH):
    result = {}
    with open(os.path.join(PATH, 'info')) as f:
        for line in f:
            k, v = line[:-1].split('\t')
            try:
                v = int(v)
            except:
                try:
                    v = float(v)
                except:
                    pass
            result[k] = v
    return result


def write_csv(filename, kwargs_list, result):
    with open(filename, 'w') as f:
        header = ['num_nodes', 'graph_id', 'sampling', 'data_size', 's', 'run_id']
        header.extend(['w_{}'.format(i) for i in range(8)])
        header.extend(['se_{}'.format(i) for i in range(8)])
        f.write(','.join(header) + '\n')
        for info, kwargs in zip(result, kwargs_list):
            row = [5000, kwargs['graph_id'], kwargs['sampling'], kwargs['n'], kwargs['s'], kwargs['i']]
            row.extend(info['weights'])
            row.extend(info['se'])
            f.write(','.join(map(str, row)) + '\n')
