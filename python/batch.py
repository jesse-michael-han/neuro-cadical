import torch
import numpy as np

def get_dims(Gs):
  """
  Args:
   Gs: a list of sparse adjacency matrices

  Returns:
    A pair of lists of row and column dimensions
  """
  return torch.transpose(torch.stack([torch.from_numpy(np.array(G.size(), dtype="int32")) for G in Gs]), 0, 1)

class Batcher:
  """
  Handles batching and unbatching for GNN.

  Is callable. When called, arguments are passed to the `batch` method.
  """
  def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), DEBUG=False):
    self.x_dims = None
    self.y_dims = None
    self.DEBUG=DEBUG
    self.device = device

  def batch(self, Gs, x_dims=None, y_dims=None):
    """
    Args:
      Gs: list of sparse adjacency matrices
      n_row_list: list of row dimensions
      n_col_list: list of column dimensions

    Returns:
      A block-diagonal adjacency matrix consisting of the members of Gs.

    Additionally sets attributes `x_dims` and `y_dims` for unbatching.
    """
    if x_dims is None:
      sizes = get_dims(Gs)
      self.x_dims = sizes[0]
      self.y_dims = sizes[1]
      
    shifts = torch.cat([torch.tensor([[0],[0]], dtype=torch.int32), sizes[:, :-1]], axis=1)
    shifts = torch.cumsum(shifts, dim=1)

    def view_helper(tsr):
      return tsr.view(tsr.size() + (1,)).to(self.device)
    
    indices = torch.cat([(G.coalesce().indices().to(self.device) + view_helper(shifts[:,k])) for k, G in enumerate(Gs)], dim=1)
    values = torch.cat([G.coalesce().values().to(indices.device) for G in Gs])
    size = torch.sum(torch.stack([torch.tensor(G.size(), dtype=torch.int32) for G in Gs]), axis=0)
    
    return torch.sparse.FloatTensor(indices=indices, values=values, size=list(size)).to(self.device)

  def unbatch(self, z, mode=None):
    try:
      assert self.x_dims is not None and self.y_dims is not None
    except AssertionError:
      raise Exception("must pass batch before trying to unbatch")
    if mode == "clause":
      return self._unbatch_clauses(z)
    elif mode == "variable":
      return self._unbatch_variables(z)
    elif mode == "literal":
      return self._unbatch_literals(z)
    else:
      raise Exception("Must specify unbatching mode (one of 'clause', 'variable', or 'literal'.)")
    
  def _unbatch_clauses(self, z):
    return torch.split(z, [int(x) for x in self.x_dims])

  def _unbatch_literals(self, z):
    return torch.split(z, self.y_dims)

  def _unbatch_variables(self, z):
    return torch.split(z, [int(x/2) for x in self.y_dims])

  def __call__(self, Gs, x_dims=None, y_dims=None):
    return self.batch(Gs, x_dims, y_dims)

