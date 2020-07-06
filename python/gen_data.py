"""
Generate data for supervised training.
"""
import time
import glob
import datetime
import tempfile
import uuid
import os
import sys
from pysat.solvers import Solver
from pysat.formula import CNF
import numpy as np
import cnfformula
import random
import subprocess
import h5py as h5
import shutil
import torch.nn as nn
from types import SimpleNamespace
import io

import util
from data_util import *
from config import *

def lemma_occ(tsr):
  n_vars = tsr.shape[0]
  result = np.zeros(shape=[n_vars])
  for idx in range(n_vars):
    result[idx] = np.sum(tsr[idx, 0, :])
  return result

def del_occ(tsr):
  n_vars = tsr.shape[0]
  result = np.zeros(shape=[n_vars])
  for idx in range(n_vars):
    result[idx] = np.sum(tsr[idx, 1, :])
  return result

class CNFDataset:
  def __init__(self):
    raise Exception("abstract method")

  def gen_formula(self):
    raise Exception("abstract method")

  def __iter__(self):
    def _gen_formula():
      while True:
        try:
          yield self.gen_formula()
        except StopIteration:
          return

    return _gen_formula()

class CNFDirDataset(CNFDataset):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.files = util.files_with_extension(self.data_dir, "cnf")
    self.file_index = 0

  def gen_formula(self):
    try:
      cnf = CNF(from_file=self.files[self.file_index])
    except IndexError:
      raise StopIteration
    self.file_index += 1
    return cnf

class Logger:
  def __init__(self):
    raise Exception("Abstract method")

  def write(self):
    raise Exception("Abstract method")

class SimpleLogger(Logger):
  def __init__(self, logfile):
    self.logfile = logfile
    util.check_make_path(logfile)

  def write(self, *args, verbose=True):
    with open(self.logfile, "a") as f:
      if verbose:
        print(f"({datetime.datetime.now()}):", *args)
      print(f"({datetime.datetime.now()}):", *args, file=f)

class DummyLogger(Logger):
  def __init__(self, verbose=False):
    self.verbose = verbose

  def write(self, *args, verbose=True, **kwargs):
    if self.verbose and verbose:
      print(*args)

def coo(fmla):
  """
  Returns sparse indices of a CNF object, as two numpy arrays.
  """
  C_result = []
  L_result = []
  for cls_idx in range(len(fmla.clauses)):
    for lit in fmla.clauses[cls_idx]:
      if lit > 0:
        lit_enc = lit - 1
      else:
        lit_enc = fmla.nv + abs(lit) - 1

      C_result.append(cls_idx)
      L_result.append(lit_enc)
  return np.array(C_result, dtype="int32"), np.array(L_result, dtype="int32")

def lbdcdl(cnf_dir, cnf, llpath, dump_dir=None, dumpfreq=50e3, timeout=None, clause_limit=1e6):
  """
  Args: CNF object, optional timeout and dump flags
  Returns: nothing
  """
  cnf_path = os.path.join(cnf_dir, str(uuid.uuid4()) + ".cnf.gz")
  cnf.to_file(cnf_path, compress_with="gzip")
  cadical_command = [CADICAL_PATH]
  cadical_command += ["-ll", llpath]
  if dump_dir is not None:
    cadical_command += ["--dump"]
    cadical_command += ["-dd", dump_dir]
    cadical_command += [f"--dumpfreq={int(dumpfreq)}"]
  if timeout is not None:
    cadical_command += ["-t", str(int(timeout))]
  if clause_limit is not None:
    cadical_command += [f"--clauselim={int(clause_limit)}"]
  cadical_command += [f"--seed={int(np.random.choice(int(10e5)))}"]
  cadical_command += [cnf_path]

  subprocess.run(cadical_command, stdout=subprocess.PIPE)

def gen_lbdp(td, cnf, is_train=True, logger=DummyLogger(verbose=True), dump_dir=None, dumpfreq=50e3, timeout=None, clause_limit=1e6):
  clause_limit = int(clause_limit)
  fmla = cnf
  counts = np.zeros(fmla.nv)  
  n_vars = fmla.nv
  n_clauses = len(fmla.clauses)
  name = str(uuid.uuid4())
  with td as td:
    llpath = os.path.join(td, name+".json")
    lbdcdl(td, fmla, llpath, dump_dir=dump_dir, dumpfreq=dumpfreq, timeout=timeout, clause_limit=clause_limit)
    with open(llpath, "r") as f:
      for idx, line in enumerate(f):
        counts[idx] = int(line.split()[1])

  C_idxs, L_idxs = coo(fmla)
  n_clauses = len(fmla.clauses)

  lbdp = LBDP(
    dp_id = name,
    is_train = np.array([is_train], dtype="bool"),
    n_vars = np.array([n_vars], dtype="int32"),
    n_clauses = np.array([n_clauses], dtype="int32"),
    C_idxs = np.array(C_idxs),
    L_idxs = np.array(L_idxs),
    glue_counts = counts
  )

  return lbdp

class CNFProcessor:
  def __init__(self, cnfdataset, tmpdir=None, use_glue_counts=False, timeout=None):
    if tmpdir is None:
      self.tmpdir = tempfile.TemporaryDirectory()
    else:
      self.tmpdir = tmpdir
    self.cnfdataset = cnfdataset
    self.use_glue_counts = use_glue_counts
    self.timeout = timeout

  def _mk_nmsdp_gen(self):
    for cnf in self.cnfdataset:
      if not self.use_glue_counts:
        nmsdp = gen_nmsdp(self.tmpdir, cnf)
      else:
        nmsdp = gen_lbdp(self.tmpdir, cnf, timeout=self.timeout)
        if np.sum(nmsdp.glue_counts) <= 50:
          continue
      self.tmpdir = tempfile.TemporaryDirectory()
      yield nmsdp

  def __iter__(self):
    return self._mk_nmsdp_gen()

  def clean(self):
    print("[CNF PROCESSOR]: CLEANING TMPDIR")
    self.tmpdir.cleanup()
    util.check_make_path(self.tmpdir.name)

  def __del__(self):
    self.clean()

