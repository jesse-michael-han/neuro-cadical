import subprocess
import time
import ray
from ray.util import ActorPool
from copy import deepcopy
import uuid
import os
import tempfile
import random
from pysat.formula import CNF
import shutil
import numpy as np
import itertools
import math

from gen_data import gen_lbdp, SimpleLogger
from util import recursively_get_files
from config import CADICAL_PATH

CLAUSE_LIMIT = int(150000)

@ray.remote
class lbdpWriter:
  def __init__(self, n_datapoints_per_file, dest):
    self.logger = SimpleLogger(os.path.join(dest, "logs", f"{str(uuid.uuid4())}-{timestamp()}.txt"))
    self.writer = DataWriter(n_datapoints_per_file, dest, out=self.logger)
    self.write_count = 0

  def write_log(self, msg):
    self.logger.write(msg)

  def write(self, x):
    self.writer.write_lbdp(x)
    self.write_count += 1
    self.write_log(f"WROTE DATAPOINT. WRITE COUNT: {self.write_count}")

  def finalize(self):
    del self.writer # force flush
    # self.logbuf.close()

  def get_write_count(self):
    return self.write_count

class Subproblems:
  def __init__(self, cnf_path, num_subproblems, random_units):
    self.cnf_path = cnf_path
    self.num_subproblems = num_subproblems
    self.random_units = random_units

  def _mk_iter(self):
    for _ in range(self.num_subproblems):
      fmla = CNF()
      fmla.from_file(self.cnf_path)
      new_units = random.sample(list(range(1, fmla.nv+1)), k=self.random_units)
      new_units = [[random.choice([1, -1]) * x] for x in new_units]
      for unit_clause in new_units:
        fmla.append(unit_clause)
      yield fmla

  def __iter__(self):
    return self._mk_iter()

class Task:
  def __init__(self, tmpdir, original_cnf_path, cnf_path, task_type, original=False):
    assert task_type == 0 or task_type == 1 or task_type == 2
    self.task_type = task_type
    self.cnf_path = cnf_path
    self.original_cnf_path = original_cnf_path
    if isinstance(tmpdir, tempfile.TemporaryDirectory):
      raise Exception("please pass tmpdir.name")
    self.tmpdir = tmpdir
    self.original = original

  def gc(self):
    if not self.original:
      if os.path.exists(self.cnf_path):
        try:
          os.remove(self.cnf_path)
        except:
          pass

class TaskManager:
  def __init__(self, cnf_path, rootdir):
    self.stage = 0
    self.tmpdir = tempfile.TemporaryDirectory(dir=rootdir)
    self.cnf_path = cnf_path
    self.tasks = [Task(self.tmpdir.name, cnf_path, cnf_path, 0, True)]
    self.waiting = 0

  def get_next_task(self):
    x = self.tasks.pop()
    self.waiting += 1
    return x

  def set_tasks(self, tasks):
    for task in tasks:
      self.tasks.append(task)
    self.waiting -= 1
    print(f"SET {len(tasks)} TASKS; NUM_TASKS = {len(self.tasks)}; STILL WAITING ON {self.waiting}")

  def _mk_iter(self):
    while True:
      try:
        yield self.get_next_task()
      except IndexError:
        return

  def __iter__(self):
    return self._mk_iter()

def simplify_cnf_path(cnf_path, CLAUSE_LIMIT):
  """
  Args
    cnf_path: absolute path to CNF file

  Returns
    an absolute path to new CNF file which is the propagated version of the old one
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    cadical_command = [CADICAL_PATH]
    cadical_command += ["-dd", tmpdir + "/"]
    cadical_command += ["-o", "foo"]
    cadical_command += ["--clauselim="+str(CLAUSE_LIMIT)]
    cadical_command += [cnf_path]
    subprocess.run(cadical_command, stdout=subprocess.DEVNULL)
    x = recursively_get_files(tmpdir, forbidden=["bz2", "xz"], exts=["cnf", "gz"])[0]
    moved = shutil.move(x, os.path.join(os.path.dirname(cnf_path), str(uuid.uuid4()) + ".cnf"))
  return moved

@ray.remote
class Worker:
  def __init__(self, writer_handle, dumpfreq=100e3, timeout=300, num_subproblems=8, random_units=5):
    self.writer_handle = writer_handle
    self.dumpfreq = dumpfreq
    self.timeout=timeout
    self.num_subproblems = num_subproblems
    self.random_units = random_units

  @ray.method(num_return_vals=1)
  def work(self, task):
    if task.task_type == 0:
      return task.original_cnf_path, self.process_task_0(task), task
    elif task.task_type == 1:
      return task.original_cnf_path, self.process_task_1(task), task
    else:
      return task.original_cnf_path, self.process_task_2(task), task

  def process_task_0(self, task):
    status, child_paths = self.extract_datapoint(task, num_subproblems=self.num_subproblems, produce_derived=True)
    tmp_tmpdir = task.tmpdir
    tmp_original_cnf_path = task.original_cnf_path
    task.gc()
    return [Task(tmp_tmpdir, tmp_original_cnf_path, path, 1 if status == 0 else 0) for path in child_paths]

  def process_task_1(self, task):
    status, child_paths = self.extract_datapoint(task, num_subproblems=self.num_subproblems, num_units=self.random_units) # extract a datapoint
    tmp_tmpdir = task.tmpdir
    tmp_original_cnf_path = task.original_cnf_path
    task.gc()
    return [Task(tmp_tmpdir, tmp_original_cnf_path, path, 2 if (not status == -1) else 0) for path in child_paths]

  def process_task_2(self, task):
    self.extract_datapoint(task)
    task.gc()
    return []

  # extract a datapoint and send it to the writer handle
  def extract_datapoint(self, task, produce_derived=False, num_subproblems=0, num_units=0):
    try:
      dump_dir = tempfile.TemporaryDirectory(dir=task.tmpdir)
    except:
      dump_dir = tempfile.TemporaryDirectory()

    with dump_dir as dump_dir_name:
      TOO_BIG_FLAG = False

      cnf = CNF()
      try:
        cnf.from_file(task.cnf_path, compressed_with="use_ext")
      except:
        try:
          new_cnf_path = simplify_cnf_path(task.cnf_path, CLAUSE_LIMIT)
          cnf.from_file(new_cnf_path, compressed_with="use_ext") # circumvent pysat DIMACS parser complaining about invalid DIMACS files
        except: # shrug
          return 1, []
        
      if len(cnf.clauses) == 0 or cnf.nv == 0:
        return 1, []
      try:
        if len(cnf.clauses) > CLAUSE_LIMIT:
          print(f"PROBLEM WITH {len(cnf.clauses)} CLAUSES TOO BIG")
          status = -1
          res = None
          TOO_BIG_FLAG = True
        else:
          res = gen_lbdp(tempfile.TemporaryDirectory(), cnf, dump_dir=(dump_dir_name + "/"), dumpfreq=self.dumpfreq, timeout=self.timeout, clause_limit=CLAUSE_LIMIT)
          glue_cutoff = 50
          if np.sum(res.glue_counts) <= glue_cutoff:
            print(f"[WARNING] PROBLEM HAS FEWER THAN {glue_cutoff} GLUE COUNTS, DISCARDING")
            return 1, []
          status = 0
      except FileNotFoundError as e:
        print("[WARNING]: FILE NOT FOUND", e)
        print("TASK TYPE: ", task.task_type)
        print("TASK ORIGINAL CNF ", task.original_cnf_path)
        status = 1
      except Exception as e:
        print("[WARNING]: EXITING GEN_LBDP DUE TO EXCEPTION", e)
        status = 1

      if status == 1:
        print(f"FORMULA {task.cnf_path} SATISFIABLE OR ERROR RAISED, DISCARDING")
        return status, []
      elif status == -1: # UNKNOWN i.e. timed out, so split the problem
        child_paths = []
        print(f"SPLITTING CNF WITH {cnf.nv} AND {len(cnf.clauses)}")
        subproblem_random_units = 3
        if TOO_BIG_FLAG:
          subproblem_random_units += math.ceil(math.log((len(cnf.clauses) - CLAUSE_LIMIT), 2))
        num_tries = 0
        while True:
          try:
            print("SPLITTING WITH RANDOM UNITS: ", subproblem_random_units)
            if num_tries > 30 or subproblem_random_units >= cnf.nv:
              print("NUM TRIES OR RANDOM UNITS EXCEEDED LIMIT, STOPPING")
              break
            for subproblem in Subproblems(task.cnf_path, num_subproblems=num_subproblems, random_units=subproblem_random_units):
              print("ENTERING SUBPROBLEM LOOP")
              subproblem_path = os.path.join(task.tmpdir, (str(uuid.uuid4()) + ".cnf.gz"))
              subproblem.to_file(subproblem_path, compress_with="gzip")
              try:
                new_path = simplify_cnf_path(subproblem_path, CLAUSE_LIMIT)
                with open(new_path, "r") as f:
                  header = f.readline().split()[2:]
                print("HEADER", header)
                n_clauses = int(header[1])
                if n_clauses > CLAUSE_LIMIT:
                  raise IndexError(f"CLAUSE LIMIT {CLAUSE_LIMIT} EXCEEDED BY N_CLAUSES {n_clauses}")
                if n_clauses == 0:
                  continue
                child_paths.append(new_path)
              except IndexError as e:
                print("SIMPLIFY CNF FOUND NO SIMPLIFIED CNF")
                print(e)
                raise IndexError
              except Exception as e:
                print("SIMPLIFY CNF PATH FAILED FOR SOME OTHER REASON")
                print(e)
                print("BAD PROBLEM: ",task.original_cnf_path)
          except IndexError:
            print("CAUGHT INDEXERROR, INCREMENTING RANDOM UNITS")
            num_tries += 1
            subproblem_random_units += (num_tries ** 2)
            continue
          break
            
        print(f"SPLIT FORMULA INTO {len(child_paths)} SUBPROBLEMS: {child_paths}")

        return status, child_paths
      else:
        self.writer_handle.write.remote(res)

        child_paths = []

        if produce_derived:
          dumps = recursively_get_files(dump_dir_name, forbidden=["bz2", "xz"], exts=["cnf", "gz"])
          print("GOT DUMPS: ", dumps)
          for cnf_path in dumps:
            try:
              moved = shutil.move(cnf_path, os.path.join(task.tmpdir, str(uuid.uuid4()) + ".cnf"))
              print(f"APPENDING DUMPED CNF: {moved}")
              child_paths.append(moved)
            except:
              print("[WARNING] SOMETHING WENT TERRIBLY WRONG")

        if num_subproblems > 0 and task.task_type == 1:
          subproblem_random_units = 3          
          while True:
            try:
              if subproblem_random_units > 15:
                break
              for subproblem in Subproblems(task.cnf_path, num_subproblems=num_subproblems, random_units=subproblem_random_units):
                subproblem_path = os.path.join(task.tmpdir, (str(uuid.uuid4()) + ".cnf.gz"))
                subproblem.to_file(subproblem_path, compress_with="gzip")
                try:
                  new_path = simplify_cnf_path(subproblem_path, CLAUSE_LIMIT)
                  child_paths.append(new_path)
                except IndexError as e:
                  print("SIMPLIFY CNF FOUND NO SIMPLIFIED CNF")
                  print(e)
                  raise IndexError
                except Exception as e:
                  print("SIMPLIFY CNF PATH FAILED FOR SOME OTHER REASON")
                  print(e)
                  print("BAD PROBLEM: ",task.original_cnf_path)
            except IndexError:
              print("CAUGHT INDEXERROR, INCREMENTING RANDOM UNITS")
              subproblem_random_units += 1
              continue
            break

      print("RETURNING CHILD PATHS", child_paths)
    return status, child_paths

def _parse_main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("cnf_dir", action="store")
  parser.add_argument("dest", action="store")
  parser.add_argument("--n-workers", dest="n_workers", action="store", type=int, default=0)
  parser.add_argument("--n-datapoints-per-file", dest="n_datapoints_per_file",type=int, action="store", default=5000)
  parser.add_argument("--dumpfreq", dest="dumpfreq", type=float, action="store", default=100e3)
  parser.add_argument("--subproblems", dest="num_subproblems", type=int, action="store", default=8)
  parser.add_argument("--random-units", dest="random_units", type=int, action="store", default=5)
  parser.add_argument("--timeout", dest="timeout", type=int, action="store", default=300)
  parser.add_argument("--rootdir", dest="rootdir", action="store", default=None)
  parser.add_argument("--n-datapoints", dest="n_datapoints", action="store", default=25000, type=int)
  opts = parser.parse_args()

  if opts.n_workers == 0:
    raise Exception("must specify positive n_workers")

  opts.dumpfreq = int(opts.dumpfreq)

  return opts

def _main(cnf_dir, n_datapoints_per_file, dest, n_workers, dumpfreq=100e3, num_subproblems=8, random_units=5, timeout=300, rootdir=None, n_datapoints=25000):
  logger = SimpleLogger(os.path.join(dest, "logs", "main_loop", f"{str(uuid.uuid4())}.txt"))
  try:
    ray.init(address='auto', redis_password='5241590000000000')
  except:
    ray.init()
  tms_dict = {cnf_path:TaskManager(cnf_path, rootdir) for cnf_path in recursively_get_files(cnf_dir, forbidden=["bz2", "xz"], exts=["cnf", "gz"])}
  logger.write("STARTING DATA GENERATION LOOP WITH", len(tms_dict.keys()), "CNFs")
  time.sleep(5)
  writer = lbdpWriter.remote(n_datapoints_per_file, dest)
  workers = [Worker.remote(writer, dumpfreq=dumpfreq, num_subproblems=num_subproblems, random_units=random_units, timeout=timeout) for _ in range(n_workers)]
  pool = ActorPool(workers)
  for tm in tms_dict.values():
    for task in tm:
      if task is not None:
        pool.submit((lambda a,v: a.work.remote(v)), task)
        logger.write(f"SUBMITTED TASK: TYPE {task.task_type}; CNF_PATH {task.cnf_path}")
  try:
    LOOP_COUNT = 0
    while any([(x.waiting > 0) for x in tms_dict.values()]):
      LOOP_COUNT += 1
      if LOOP_COUNT % 100 == 0:
        if ray.get(writer.get_write_count.remote()) > n_datapoints:
          print(f"NUMBER OF WRITES EXCEEDS N_DATAPOINTS={n_datapoints}, STOPPING")
          break
      cnf_path, tasks, original_task = pool.get_next_unordered()
      logger.write(f"GOT TASK RESULT (TYPE {original_task.task_type})")
      if original_task.task_type == 0 and len(tasks) == 0:
        logger.write("WARNING: Task", original_task.cnf_path, "returned no subtasks.")
      tm = tms_dict[cnf_path]
      tm.set_tasks(tasks)
      if (tm.waiting == 0) and len(tm.tasks) == 0:
        print(f"ROOT TASK {tm.cnf_path} FINISHED")
        tms_dict.pop(cnf_path)
        print("POPPED FROM TMS_DICT")
        try:
          shutil.rmtree(tms.tmpdir.name)
          os.makedirs(tms.tmpdir.name)
          time.sleep(1)
        except:
          pass
      else:
        for task in tm:
          if task.task_type == 0:
            logger.write(f"SUBMITTING SUBPROBLEM (TIMEOUT) TASK")
          elif task.task_type == 1:
            logger.write(f"SUBMITTING DERIVED FORMULA")
          elif task.task_type == 2:
            logger.write(f"SUBMITTING DERIVED SUBFORMULA")
          pool.submit((lambda a,v: a.work.remote(v)), task)
        logger.write(f"SUBMITTED {len(tasks)} NEW TASKS")
  finally:
    del writer

if __name__ == "__main__":
  _main(**vars(_parse_main()))
