import random
import re
import shutil
import os
import h5py
import collections
import numpy as np
import uuid
import tempfile
import sys
import itertools
from math import ceil
import torch.utils.data as td

from util import *

NMSDP = collections.namedtuple( # all fields besides dp_id must be numpy arrays
  "NMSDP",
  ["dp_id",
   "is_train",
   "n_vars",
   "n_clauses",
   "C_idxs",
   "L_idxs",
    "core_var_mask",
   "core_clause_mask",
   "var_lemma_counts"]
)

LBDP = collections.namedtuple( # all fields besides dp_id must be numpy arrays
  "LBDP",
  ["dp_id",
   "is_train",
   "n_vars",
   "n_clauses",
   "C_idxs",
   "L_idxs",
   "glue_counts"]
)

def serialize_lbdp(lbdp, f):
  return serialize_nmsdp(lbdp, f)

def deserialize_lbdp(grp, dp_id):
  return LBDP(
    dp_id = dp_id,
    is_train = grp["is_train"][()],
    n_vars = grp["n_vars"][()],
    n_clauses = grp["n_clauses"][()],
    C_idxs = grp["C_idxs"][()],
    L_idxs = grp["L_idxs"][()],
    glue_counts = grp["glue_counts"][()]
  )

def serialize_nmsdp(nmsdp, f):
  dp_id = nmsdp.dp_id
  grp = f.create_group(dp_id)
  for key in nmsdp._fields[1:]:
    try:
      grp.create_dataset(key, data=getattr(nmsdp, key), compression="gzip")
    except TypeError:
      print("BAD KEY", key)
      raise Exception

def deserialize_nmsdp(grp, dp_id):
  return NMSDP(
    dp_id = dp_id,
    is_train = grp["is_train"][()],
    n_vars = grp["n_vars"][()],
    n_clauses = grp["n_clauses"][()],
    C_idxs = grp["C_idxs"][()],
    L_idxs = grp["L_idxs"][()],
    core_var_mask = grp["core_var_mask"][()],
    core_clause_mask = grp["core_clause_mask"][()],
    var_lemma_counts = grp["var_lemma_counts"][()]
  )

class DataWriter:
  def __init__(self, n_datapoints_per_file, dest, out=sys.stdout):
    self.n_datapoints_per_file = n_datapoints_per_file
    self.dest = dest
    self.TOTAL_WRITE_COUNT = 0
    self.FILE_COUNT = 0
    self.log_stream = out
    self.tmpdir = tempfile.TemporaryDirectory()
    self.prepare_next_file() # sets current file handle
    if not os.path.exists(self.dest):
      os.makedirs(dest)

  def prepare_next_file(self):
    print("Preparing next file.", file=self.log_stream)
    self.FILE_COUNT += 1
    self.FILE_WRITE_COUNT = 0
    self.outfile = f"file{self.FILE_COUNT}_{str(uuid.uuid4())}.h5"
    self.outfile_path = os.path.join(self.tmpdir.name, self.outfile)
    self.current_file_handle = h5py.File(self.outfile_path, "a")

  def finish_file(self):
    print("Finishing and moving file.", file=self.log_stream)
    if self.FILE_WRITE_COUNT > 0:
      self.current_file_handle.flush()
      self.current_file_handle.close()
      shutil.move(self.outfile_path, os.path.join(self.dest, self.outfile))

  def write_nmsdp(self, nmsdp):
    print(f"FILE WRITE COUNT: {self.FILE_WRITE_COUNT}", file=self.log_stream)
    print(f"FILE COUNT: {self.FILE_COUNT}", file=self.log_stream)
    if self.FILE_WRITE_COUNT >= self.n_datapoints_per_file:
      self.finish_file()
      self.prepare_next_file()
    serialize_nmsdp(nmsdp, self.current_file_handle)
    self.TOTAL_WRITE_COUNT += 1
    self.FILE_WRITE_COUNT += 1

  def write_lbdp(self, lbdp):
    print(f"FILE WRITE COUNT: {self.FILE_WRITE_COUNT}", file=self.log_stream)
    print(f"FILE COUNT: {self.FILE_COUNT}", file=self.log_stream)
    if self.FILE_WRITE_COUNT >= self.n_datapoints_per_file:
      self.finish_file()
      self.prepare_next_file()
    serialize_lbdp(lbdp, self.current_file_handle)
    self.TOTAL_WRITE_COUNT += 1
    self.FILE_WRITE_COUNT += 1

  def __del__(self):
    print("Finalizing due to garbage collection.", file=self.log_stream)
    self.finish_file()

class ListSharder: # responsible for cycling and sharding
  def __init__(self, xs, n_shards):
    self.shard_size = ceil(len(xs)/n_shards)
    self.n_shards = n_shards
    self.xs = xs
    random.shuffle(self.xs)

  @property
  def xs_iter(self):
    return itertools.cycle(self.xs)

  def get_shard(self, index):
    start = index * self.shard_size
    stop = (index + 1) * self.shard_size
    return list(itertools.islice(self.xs_iter, start, stop))

def batch_iterator(it, batch_size):
  while True:
    count = 0
    result = []
    try:
      while count < batch_size:
        result.append(next(it))
        count += 1
      yield result
    except StopIteration:
      # if len(result) == 0:
      #   return
      # else:
      #   yield result
      return # note(jesse, March 04 2020, 07:44 PM): drop the last batch for now


class BatchedIterable(td.IterableDataset):
  def __init__(self, it, batch_size):
    """
    Args:
      it: an iterable
      batch_size: an integer
    """
    super(BatchedIterable, self).__init__()
    self.it = it
    self.batch_size = batch_size
    
  def __iter__(self):
    return batch_iterator(self.it.__iter__(), self.batch_size)

def shuffle_with_buffer(it0, buf_size):
  buf = []
  it = it0.__iter__()
  FLAG1 = False
  while True:
    if not FLAG1:
      if len(buf) < buf_size:
        try:
          next_val = next(it)
        except StopIteration:
          return
        buf.append(next_val)
        continue
      else:
        FLAG1 = True
        continue
    random.shuffle(buf)
    for x in buf:
      yield x
    FLAG1=False
    buf=[]
    continue

class H5Dataset(td.IterableDataset):
  """
  Dataset which yields either single NMSDPs or lists of NMSDPs (depending on if batch_size is None or a positive integer.)
  """
  def __init__(self, data_dir, batch_size=None):
    super(H5Dataset, self).__init__()
    self.data_dir = data_dir
    self.files = files_with_extension(self.data_dir, "h5")
    self.shuffle_files()
    self.batch_size = batch_size

  def shuffle_files(self):
    random.shuffle(self.files)

  def _mk_iter(self):
    for f in self.files:
      with h5py.File(f, "r") as f:
        for dp_id in f:
          yield deserialize_lbdp(f[dp_id], dp_id)

  def __iter__(self):
    if self.batch_size is None:
      return self._mk_iter()
    else:
      return batch_iterator(self._mk_iter(), self.batch_size)

  def dist_configure(self, rank, size):
    ls = ListSharder(self.files, size)
    self.files = ls.get_shard(rank)

def h5_worker_init_fn(worker_id):
    worker_info = td.get_worker_info()
    ls = ListSharder(worker_info.dataset.files, worker_info.num_workers)
    worker_info.dataset.files = ls.get_shard(worker_info.id) # shard files only
    random.shuffle(worker_info.dataset.files)
    print(f"[DATALOADER] STARTING WORKER {worker_id} WITH SHARD OF {len(worker_info.dataset.files)} FILES")

def mk_H5DataLoader(data_dir, batch_size, num_workers):
  """
  Helper function for constructing a parallelized H5DataLoader which shards the files in `data_dir` among `num_workers` workers.

  `batch_size` is used to wrap each copy of the `H5Dataset` with a `BatchedIterable` returning lists of NMSDPs.

  Since `DataLoader` automatically tries to pack batches into a tensor, we construct the DataLoader with `batch_size=1`, moving the batching into the Dataset itself.

  (WARNING: this is an abuse of the API.)
  """
  h5d = H5Dataset(data_dir, batch_size=batch_size)

  return td.DataLoader(h5d, batch_size=1, num_workers=num_workers, worker_init_fn=h5_worker_init_fn, pin_memory=True)
