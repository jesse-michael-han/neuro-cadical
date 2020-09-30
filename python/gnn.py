import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from batch import Batcher
import util

def decode_activation(activation):
  if activation == "relu":
    return nn.ReLU
  elif activation == "leaky_relu":
    return nn.LeakyReLU
  elif activation == "relu6":
    return nn.ReLU6
  elif activation == "elu":
    return nn.ELU
  else:
    raise Exception("unsupported activation")

class BasicMLP(nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim, activation, bias_at_end=True, p_dropout=0.1, **kwargs):
    super(BasicMLP, self).__init__(**kwargs)
    layers = []
    for k in range(len(hidden_dims) + 1):
      if k == 0:
        d_in = input_dim
      else:
        d_in = hidden_dims[k-1]

      if k == len(hidden_dims):
        d_out = output_dim
      else:
        d_out = hidden_dims[k]

      layers.append(nn.Linear(in_features=d_in, out_features=d_out, bias=(True if ((k == len(hidden_dims) and bias_at_end) or k < len(hidden_dims)) else False)))

      if not (k == len(hidden_dims)):
        layers.append(decode_activation(activation)())
        layers.append(nn.Dropout(p_dropout))

    self.main = nn.Sequential(*layers)

  def forward(self, z):
    return self.main(z)

defaultGNN1Cfg = {
  "clause_dim":64,
  "lit_dim":16,
  "n_hops":2,
  "n_layers_C_update":0,
  "n_layers_L_update":0,
  "n_layers_score":1,
  "activation":"relu"
}

def swap_polarity(G):
  indices = G.coalesce().indices()
  values = G.coalesce().values()
  size = G.size()

  pivot = size[1]/2 - 0.5

  indices[1] = (2.0 * pivot) - indices[1]

  return torch.sparse.FloatTensor(
    indices=indices,
    values=values,
    size=size)

def flop(L_logits):
  return torch.cat([L_logits[0:int(L_logits.size()[0]/2)], L_logits[int(L_logits.size()[0]/2):]], dim=1)

def flip(L_logits):
  return torch.cat([L_logits[int(L_logits.size()[0]/2):], L_logits[0:int(L_logits.size()[0]/2)]], dim=0)

class GNN1(nn.Module):
  def __init__(self,
               clause_dim,
               lit_dim,
               n_hops,
               n_layers_C_update,
               n_layers_L_update,
               n_layers_score,
               activation,
               average_pool=False,
               normalize=True,
               **kwargs):
    super(GNN1, self).__init__(**kwargs)
    self.L_layer_norm = nn.LayerNorm(lit_dim)
    self.L_init = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty([1, lit_dim])), requires_grad=True)
    self.C_update = BasicMLP(input_dim=2*lit_dim,
                             hidden_dims=[2*lit_dim for _ in range(n_layers_C_update)],
                             output_dim=clause_dim, activation=activation, p_dropout=0.05)
    self.L_update = BasicMLP(input_dim=(clause_dim),
                             hidden_dims=[clause_dim for _ in range(n_layers_L_update)],
                             output_dim=lit_dim, activation=activation, p_dropout=0.05)
    self.V_score_drat = BasicMLP(input_dim=2*lit_dim, hidden_dims=[2*lit_dim for _ in range(n_layers_score)],
                                 output_dim=1, activation=activation, bias_at_end=True, p_dropout=0.15)
    self.V_score_core = BasicMLP(input_dim=2*lit_dim, hidden_dims=[2*lit_dim for _ in range(n_layers_score)],
                                 output_dim=1, activation=activation, p_dropout=0.05)
    self.C_score_core = BasicMLP(input_dim=clause_dim, hidden_dims=[clause_dim for _ in range(n_layers_score)],
                                 output_dim=1, activation=activation, p_dropout=0.05)

    self.n_hops = n_hops
    self.lit_dim = lit_dim
    self.clause_dim = clause_dim
    self.average_pool = average_pool
    self.normalize = normalize
    if not self.normalize:
      self.C_layer_norm = nn.LayerNorm(clause_dim)

  def forward(self, G):
    n_clauses, n_lits = G.size()
    n_vars = n_lits/2
    L = self.L_init.repeat(n_lits, 1)
    if not (G.device == L.device):
      L = L.to(G.device)
      
    for T in range(self.n_hops):
      L_flip = torch.cat([L[int(L.size()[0]/2):], L[0:int(L.size()[0]/2)]], dim=0)
      if self.average_pool:
        C_pre_msg = torch.cat([L, L_flip, torch.ones(G.size()[1],1, dtype=torch.float32, device=G.device)], dim=1)
      else:
        C_pre_msg = torch.cat([L, L_flip], dim=1)
      C_msg = torch.sparse.mm(G, C_pre_msg)

      if self.average_pool:
        
        C_neighbor_counts = C_msg[:,-1:]

        C_msg = C_msg[:, :-1]

        C_msg = C_msg/torch.max(C_neighbor_counts, torch.ones(C_neighbor_counts.size()[0], C_neighbor_counts.size()[1], device=G.device))

      
      C = self.C_update(C_msg)
      if self.normalize:
        C = C - C.mean(dim=0)
        C = C/(C.std(dim=0) + 1e-10)
      else:
        C = self.C_layer_norm(C)
      if self.average_pool:
        L_pre_msg = torch.cat([C, torch.ones(G.size()[0],1, dtype=torch.float32,device=G.device)], dim=1)
      else:
        L_pre_msg = C
      L_msg = torch.sparse.mm(G.t(), L_pre_msg)
      if self.average_pool:
        L_neighbor_counts = L_msg[:,-1:]
        L_msg = L_msg[:,:-1]
        L_msg = L_msg/torch.max(L_neighbor_counts, torch.ones(L_neighbor_counts.size()[0],L_neighbor_counts.size()[1], device=G.device))
      L = self.L_update(L_msg) + (0.1 * L)
      L = self.L_layer_norm(L)
    V = torch.cat([L[0:int(L.size()[0]/2)], L[int(L.size()[0]/2):]], dim=1)
    return self.V_score_drat(V), self.V_score_core(V), self.C_score_core(C)

class GNN1_drat(nn.Module): # deploy the drat head only
  def __init__(self,
               clause_dim,
               lit_dim,
               n_hops,
               n_layers_C_update,
               n_layers_L_update,
               n_layers_score,
               activation,
               average_pool=False,
               normalize=True,
               **kwargs):
    super(GNN1_drat, self).__init__(**kwargs)
    self.L_layer_norm = nn.LayerNorm(lit_dim)
    self.L_init = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty([1, lit_dim])), requires_grad=True)

    self.C_update = BasicMLP(input_dim=2*lit_dim,
                             hidden_dims=[2*lit_dim for _ in range(n_layers_C_update)],
                             output_dim=clause_dim, activation=activation)
    self.L_update = BasicMLP(input_dim=(clause_dim),
                             hidden_dims=[clause_dim for _ in range(n_layers_L_update)],
                             output_dim=lit_dim, activation=activation)

    self.V_score_drat = BasicMLP(input_dim=2*lit_dim, hidden_dims=[2*lit_dim for _ in range(n_layers_score)],
                                 output_dim=1, activation=activation, bias_at_end=True)

    self.n_hops = n_hops
    self.lit_dim = lit_dim
    self.clause_dim = clause_dim
    self.average_pool = average_pool
    self.normalize = normalize

  def forward(self, G):
    n_clauses, n_lits = G.shape
    n_vars = n_lits/2
    L = self.L_init.repeat(n_lits, 1)
    if not (G.device == L.device):
      L = L.to(G.device)

    for T in range(self.n_hops):
      L_flip = torch.cat([L[int(L.size()[0]/2):], L[0:int(L.size()[0]/2)]], dim=0)      
      if self.average_pool:
        C_pre_msg = torch.cat([L, L_flip, torch.ones(G.size()[1],1, dtype=torch.float32, device=G.device)], dim=1)
      else:
        C_pre_msg = torch.cat([L, L_flip], dim=1)
      C_msg = torch.sparse.mm(G, C_pre_msg)

      if self.average_pool:
        C_neighbor_counts = C_msg[:,-1:]
        C_msg = C_msg[:, :-1]
        C_msg = C_msg/torch.max(C_neighbor_counts, torch.ones(C_neighbor_counts.size()[0], C_neighbor_counts.size()[1], device=G.device))

      C = self.C_update(C_msg)
      if self.normalize:
        C = C - C.mean(dim=0)
        C = C/(C.std(dim=0) + 1e-10)
      if self.average_pool:
        L_pre_msg = torch.cat([C, torch.ones(G.size()[0],1, dtype=torch.float32,device=G.device)], dim=1)
      else:
        L_pre_msg = C
      L_msg = torch.sparse.mm(G.t(), L_pre_msg)
      if self.average_pool:
        L_neighbor_counts = L_msg[:,-1:]
        L_msg = L_msg[:,:-1]
        L_msg = L_msg/torch.max(L_neighbor_counts, torch.ones(L_neighbor_counts.size()[0],L_neighbor_counts.size()[1], device=G.device))
      L = self.L_update(L_msg) + (0.1 * L)
      L = self.L_layer_norm(L)
    V = torch.cat([L[0:int(L.size()[0]/2)], L[int(L.size()[0]/2):]], dim=1)
    return self.V_score_drat(V)


class rl_GNN1(nn.Module):
  def __init__(self,
               clause_dim,
               lit_dim,
               n_hops,
               n_layers_C_update,
               n_layers_L_update,
               n_layers_score,
               activation,
               average_pool=False,
               normalize=True,
               **kwargs):
    super(rl_GNN1, self).__init__(**kwargs)
    self.L_layer_norm = nn.LayerNorm(lit_dim)
    self.L_init = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty([1, lit_dim])), requires_grad=True)

    self.C_update = BasicMLP(input_dim=2*lit_dim,
                             hidden_dims=[2*lit_dim for _ in range(n_layers_C_update)],
                             output_dim=clause_dim, activation=activation, p_dropout=0.05)
    self.L_update = BasicMLP(input_dim=(clause_dim),
                             hidden_dims=[clause_dim for _ in range(n_layers_L_update)],
                             output_dim=lit_dim, activation=activation, p_dropout=0.05)

    self.V_score = BasicMLP(input_dim=2*lit_dim, hidden_dims=[2*lit_dim for _ in range(n_layers_score)],
                                 output_dim=1, activation=activation, bias_at_end=True, p_dropout=0.15)
    self.V_vote = BasicMLP(input_dim=2*lit_dim, hidden_dims=[2*lit_dim for _ in range(n_layers_score)],
                                 output_dim=1, activation=activation, bias_at_end=True, p_dropout=0.15)

    self.n_hops = n_hops
    self.lit_dim = lit_dim
    self.clause_dim = clause_dim
    self.average_pool = average_pool
    self.normalize = normalize
    if not self.normalize:
      self.C_layer_norm = nn.LayerNorm(clause_dim)

  def forward(self, G):
    n_clauses, n_lits = G.size()
    n_vars = n_lits/2
    L = self.L_init.repeat(n_lits, 1)
    if not (G.device == L.device):
      L = L.to(G.device)

    for T in range(self.n_hops):
      L_flip = torch.cat([L[int(L.size()[0]/2):], L[0:int(L.size()[0]/2)]], dim=0)
      if self.average_pool:
        C_pre_msg = torch.cat([L, L_flip, torch.ones(G.size()[1],1, dtype=torch.float32, device=G.device)], dim=1)
      else:
        C_pre_msg = torch.cat([L, L_flip], dim=1)
      C_msg = torch.sparse.mm(G, C_pre_msg)

      if self.average_pool:
        C_neighbor_counts = C_msg[:,-1:]

        C_msg = C_msg[:, :-1]

        C_msg = C_msg/torch.max(C_neighbor_counts, torch.ones(C_neighbor_counts.size()[0], C_neighbor_counts.size()[1], device=G.device))

      C = self.C_update(C_msg)
      if self.normalize:
        C = C - C.mean(dim=0)
        C = C/(C.std(dim=0) + 1e-10)
      else:
        C = self.C_layer_norm(C)
      if self.average_pool:
        L_pre_msg = torch.cat([C, torch.ones(G.size()[0],1, dtype=torch.float32,device=G.device)], dim=1)
      else:
        L_pre_msg = C
      L_msg = torch.sparse.mm(G.t(), L_pre_msg)
      if self.average_pool:
        L_neighbor_counts = L_msg[:,-1:]
        L_msg = L_msg[:,:-1]
        L_msg = L_msg/torch.max(L_neighbor_counts, torch.ones(L_neighbor_counts.size()[0],L_neighbor_counts.size()[1], device=G.device))
      L = self.L_update(L_msg) + (0.1 * L)
      L = self.L_layer_norm(L)


    V = torch.cat([L[0:int(L.size()[0]/2)], L[int(L.size()[0]/2):]], dim=1)

    return (self.V_score(V), self.V_vote(V)) # return policy logits and value logits before averaging (for unbatching)
