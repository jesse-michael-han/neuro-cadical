import re
import tempfile
from pathlib import Path
import os
import uuid
import json
import subprocess
import time
import psutil
import random
import datetime
from subprocess import PIPE
import ray
from ray.util import ActorPool
from ray.util.multiprocessing import Pool

import util
from util import recursively_get_files
from config import CADICAL_PATH

class DummyWorker:
  def __init__(self):
    pass

  def run_fn(self, f, *args, **kwargs):
    return f(*args, **kwargs)

def cadical_fn(cnf_path, refocus=False, query_interval=None, model=None, verbose=False, refocus_scale=1e7, random_refocus=False, cpu_lim=None, irrlim=10e6, seed=None, refocus_init_time=60, refocus_base=5000, refocus_exp=1, refocus_ceil=50000, config=None, gpu=False, refocus_glue_sucks=False, refocus_glue_sucks_margin=20, refocus_reluctant=False, refocus_rebump=False, refocus_restart=True, elim_rel_eff=1000, subsume_rel_eff=1000, rephase=True, stabilize_only=False, walk=True, **kwargs):
  query_interval = int(query_interval) if query_interval is not None else 50000
  cpu_lim = int(cpu_lim)
  refocus_scale = int(refocus_scale)
  irrlim = int(irrlim)
  refocus_ceil = int(refocus_ceil)
  refocus_glue_sucks_margin = int(refocus_glue_sucks_margin)

  res = str(uuid.uuid4())
  with tempfile.TemporaryDirectory() as tmpdir:
    so_path = os.path.join(tmpdir, res + ".json")
    cadical_path = CADICAL_PATH

    cadical_command = [cadical_path]
    cadical_command += ["-so", so_path]
    if cpu_lim is not None:
      cadical_command += ["-t", str(cpu_lim)]
    if model is not None:
      cadical_command += ["-model", model]
    if refocus:
      cadical_command += ["--refocus"]
    else:
      cadical_command += ["--no-refocus"]
    if query_interval is not None:
      cadical_command += [f"--queryinterval={query_interval}"]
    cadical_command += [f"--refocusscale={refocus_scale}"]
    if random_refocus:
      cadical_command += ["--randomrefocus"]

    cadical_command += [f"--irrlim={irrlim}"]

    if seed is not None:
      cadical_command += [f"--seed={int(seed)}"]

    cadical_command += [f"--refocusinittime={int(refocus_init_time)}"]
    cadical_command += [f"--refocusdecaybase={int(refocus_base)}"]
    cadical_command += [f"--refocusdecayexp={int(refocus_exp)}"]
    cadical_command += [f"--refocusceil={refocus_ceil}"]
    if config == "sat":
      cadical_command += ["--sat"]
    elif config == "unsat":
      cadical_command += ["--unsat"]
    else:
      pass
    if elim_rel_eff is not None:
      cadical_command += [f"--elimreleff={int(elim_rel_eff)}"]
    if subsume_rel_eff is not None:
      cadical_command += [f"--subsumereleff={int(subsume_rel_eff)}"]

    cadical_command += [f"--rephase="+("true" if rephase else "false")]
    cadical_command += [f"--stabilizeonly="+("true" if stabilize_only else "false")]
    cadical_command += [f"--walk="+("true" if walk else "false")]

    if gpu:
      cadical_command += ["--gpu"]

    if refocus_glue_sucks:
      cadical_command += ["--refocusgluesucks"]
      cadical_command += [f"--refocusgluesucksmargin={refocus_glue_sucks_margin}"]

    if refocus_reluctant:
      cadical_command += ["--refocusreluctant"]

    if refocus_rebump:
      cadical_command += ["--refocusrebump"]

    if refocus_restart:
      cadical_command += ["--refocusrestart"]

    cadical_command += [cnf_path]

    start = time.time()
    result = subprocess.run(cadical_command, stdout=PIPE)
    elapsed = time.time() - start

    try:
      with open(so_path, "r") as f:
        try:
          result_dict = json.load(f)
        except json.decoder.JSONDecodeError:
          msg = ""
          for x in f.readlines():
            print("HEWWO", x)
            msg += x
          print("BAD CNF", cnf_path)
          raise Exception
    except FileNotFoundError:
      result_dict = {"errored_out":True}

    if verbose:
      print("COMMAND: ", " ".join(cadical_command))

  result_dict["wall_time"] = elapsed
  result_dict["instance_name"] = Path(cnf_path).name # just use the entire file name

  if result_dict.get("errored_out", False):
    print("ERRORED OUT: ", result_dict["instance_name"])
    verbose=True

  if verbose:
    for line in result.stdout.decode("utf8").split("\n"):
      print(line)

  return result_dict

class StatHolder:
  def __init__(self, program_name, prog_args, prog_alias, benchmark, m_options=dict()):
    self.preamble = {
        "program" : program_name,
        "prog_args" : prog_args,
        "prog_alias" : prog_alias,
        "benchmark" : benchmark
      }

    self.start = time.time()

    self.solve_count = 0

    self.preamble.update(m_options)

    self.stats = dict()

  def store_result(self, instance_name, result_dict):
    if result_dict.get("errored_out", False):
      self.stats[instance_name] = {"status" : False,
                                   "rtime" : 0,
                                   "decisions" : 0,
                                   "propagations" : 0,
                                   "mem_used"    : 0,
                                   "result" : None,
                                   "conflicts" : 0,
                                   "glr"   : 0,
                                   "errored_out" : True
      }
    else:
      try:
        glr = float(result_dict["conflicts"]/result_dict["decisions"])
      except ZeroDivisionError:
        glr = 0
      self.stats[instance_name] = {"status" : result_dict["result"] is not None,
                                   "rtime" : result_dict["cpu_time"],
                                   "decisions" : result_dict["decisions"],
                                   "propagations" : result_dict["propagations"],
                                   "mem_used"    : result_dict["mem_used"],
                                   "result" : result_dict["result"],
                                   "conflicts" : result_dict["conflicts"],
                                   "glr"   : glr,
                                   "num_queries" : result_dict["num_queries"],
                                   "random_seed" : result_dict.get("random_seed", None),
                                   "avg_refocus_time" : result_dict.get("avg_refocus_time", 0),
                                   "avg_glue" : result_dict.get("avg_glue", 0),
                                   "oom_count" : result_dict.get("oom_count", 0)
      }

    if self.stats[instance_name]["status"]:
      self.solve_count += 1
      elapsed = time.time() - self.start
      print(f"SOLVED {self.solve_count} INSTANCES WITHIN {elapsed}s")

  def get_dict(self):
    result = {"preamble" : self.preamble, "stats" : self.stats}
    return result

  def dump_json(self, path):
    util.check_make_path(path)
    with open(path, "w") as f:
      f.write(json.dumps(self.get_dict(), indent=2))

def test_harness(cnf_dir, m_options, statholder_options, out_path, pool, exts=["cnf", "gz"], forbidden=[], solver="cadical"):
  if solver == "cadical":
    solver_fn = cadical_fn
  else:
    raise Exception("unsupported solver!")
  files = recursively_get_files(cnf_dir, exts, forbidden=forbidden)

  def work(path):
    return solver_fn(path, **m_options)

  time.sleep(15)

  result_handles = pool.map_unordered((lambda a,v: a.run_fn.remote(work, v)), files)

  sh = StatHolder(**statholder_options, m_options=m_options)

  for result in result_handles:
    sh.store_result(result["instance_name"], result)
    sh.dump_json(out_path)

  sh.dump_json(out_path)
  print(f"({datetime.datetime.now()}): DONE -- {out_path}")
  time.sleep(15)

def pin_to_core(worker):
  """
  Worker is a handle to a remote Ray actor with a `run_fn` method.
  """
  resource_ids = ray.get(worker.run_fn.remote((lambda: ray.get_resource_ids())))
  cpu_idx = resource_ids["CPU"][0][0]
  ray.get(worker.run_fn.remote((lambda: psutil.Process().cpu_affinity([cpu_idx]))))
