from copy import deepcopy
from pathlib import Path
import itertools
import time
import numpy as np
import tempfile
import os

from solver import *
from config import TORCHSCRIPT_MODEL_PATH
from deploy_model import *

SVCOMP_PATH = ""
SATCOMP18_PATH = ""

BENCHMARKS = [SVCOMP_PATH, SATCOMP18_PATH]

def mk_cadical_options():
  RESULT = []
  refocuses = [True]
  cpu_lims = [5000]
  query_intervals = [50000]
  refocus_bases = [1000]
  refocus_exps = [2]
  refocus_ceils = [250000]
  refocus_init_times = [15]
  refocus_glue_suckss = [False]
  refocus_glue_sucks_margins = [20]
  refocus_reluctants = [False]

  refocus_scales = [10000.0]
  irrlims = [10e6]

  BASELINE_OPTIONS = []
  BASELINE_OPTIONS.append({
    "refocus":False,
    "cpu_lim":5000,
    "config":"sat"
  })

  for bm, clim, qi, rs, irrlim, refocus_base, refocus_exp, refocus_init_time, refocus_ceil, glue_sucks, glue_sucks_margin, refocus_reluctant in itertools.product(refocuses, cpu_lims, query_intervals, refocus_scales, irrlims, refocus_bases, refocus_exps, refocus_init_times, refocus_ceils, refocus_glue_suckss, refocus_glue_sucks_margins, refocus_reluctants):
    mopts = {
      "refocus":bm,
      "cpu_lim":clim,
      "query_interval":qi,
      "refocus_scale":rs,
      "irrlim":irrlim,
      "refocus_base":refocus_base,
      "refocus_exp":refocus_exp,
      "refocus_init_time":refocus_init_time,
      "refocus_ceil":refocus_ceil,
      "config":"sat",
      "refocus_glue_sucks":glue_sucks,
      "refocus_glue_sucks_margin":glue_sucks_margin,
      "refocus_reluctant":refocus_reluctant
    }

    RESULT.append(mopts)
    mopts2 = deepcopy(mopts)
    mopts2["random_refocus"] = True

    BASELINE_OPTIONS.append(mopts2)

  return RESULT, BASELINE_OPTIONS

CADICAL_SOLVER_OPTIONS, CADICAL_BASELINE_OPTIONS = mk_cadical_options()

def get_name(path):
  ckpt_name = Path(path).stem
  model_name = Path(os.path.dirname(path)).stem
  return model_name + "-"  + ckpt_name

def _parse_main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--benchmarks", action="store", dest="benchmarks", nargs="*", default=None)
  parser.add_argument("cfg", action="store", type=str)
  parser.add_argument("ckpt", action="store", type=str)
  parser.add_argument("--n-workers", dest="n_workers", action="store", type=int)
  parser.add_argument("--log-dir", dest="log_dir", action="store", default=os.path.join("home", "ubuntu", "efs", "solver_eval"))
  parser.add_argument("--baseline", action="store_true")
  parser.add_argument("--no-neuro", dest="no_neuro", action="store_true")
  parser.add_argument("--timeout",type=int,dest="timeout", action="store", default=5000)
  parser.add_argument("--mem-per-worker", type=int, dest="mem_per_worker", action="store", default=8)
  parser.add_argument("--glue-sucks", dest="glue_sucks", action="store_true")
  parser.add_argument("--glue-sucks-margin", dest="glue_sucks_margin", action="store", default=20)
  parser.add_argument("--config", dest="config", default=None, action="store")
  parser.add_argument("--refocus-reluctant", dest="refocus_reluctant", action="store_true")
  parser.add_argument("--refocus-rebump", dest="refocus_rebump", action="store_true")
  parser.add_argument("--refocus-restart", dest="refocus_restart", action="store_true")  
  parser.add_argument("--query-interval", dest="query_interval", action="store", type=float, default=50000)
  parser.add_argument("--refocus-ceil", dest="refocus_ceil", action="store", type=float, default=250000)
  parser.add_argument("--refocus-init-time", dest="refocus_init_time", action="store", type=int, default=15)
  parser.add_argument("--refocus-base", dest="refocus_base", action="store", type=int, default=1000)
  parser.add_argument("--refocus-exp", dest="refocus_exp", action="store", type=int, default=2)
  parser.add_argument("--elim-rel-eff", dest="elim_rel_eff", action="store", default=1000, type=int)
  parser.add_argument("--subsume-rel-eff", dest="subsume_rel_eff", action="store", default=1000, type=int)
  parser.add_argument("--rephase", dest="rephase", action="store", type=int, default=1)
  parser.add_argument("--stabilize-only", dest="stabilize_only", default=0, type=int)
  parser.add_argument("--walk", dest="walk", action="store", default=1, type=int)
  opts = parser.parse_args()
  try:
    assert opts.config == "sat" or opts.config == "unsat" or opts.config is None
  except AssertionError as e:
    print(e)
    raise Exception("must specify --config as sat, unsat, or None")
  return opts

def is_baseline(mopts):
  return mopts.get("random_refocus", False) or mopts.get("branch_mode", 1) == 0 or (not mopts.get("refocus", True))

def _main(cfg, ckpt, n_workers, log_dir, benchmarks, baseline, no_neuro, timeout, mem_per_worker=None, glue_sucks=False, config=None, refocus_reluctant=False, refocus_rebump=False, query_interval=50000, refocus_ceil=250000, refocus_restart=True, glue_sucks_margin=20, refocus_init_time=15, refocus_base=1000, refocus_exp=2, elim_rel_eff=1000, subsume_rel_eff=1000, rephase=True, stabilize_only=False, walk=True):
  seed = np.random.choice(int(1e4)) # use fixed seed throughout
  if mem_per_worker == 0:
    mem_per_worker = None
  if benchmarks is None:
    benchmarks = BENCHMARKS
  name = get_name(ckpt)
  DEPLOY_PATH = "/tmp/"
  with tempfile.TemporaryDirectory(dir=DEPLOY_PATH) as tmpdir:
    model_drat_path = os.path.join(tmpdir, name + "_drat.pt")
    deploy_GNN1_drat(model_cfg_path=cfg, ckpt_path=ckpt, save_path=model_drat_path)

    for cnfdir in benchmarks:
      OPTIONS = CADICAL_SOLVER_OPTIONS if not baseline else CADICAL_SOLVER_OPTIONS + CADICAL_BASELINE_OPTIONS

      if no_neuro:
        OPTIONS = [{
          "refocus":False,
          "cpu_lim":timeout,
          "config":config
        }]

      TMPOPTIONS = []
      for mopt in OPTIONS:
        mopt["refocus_base"] = refocus_base
        mopt["refocus_exp"] = refocus_exp
        mopt["refocus_restart"] = refocus_restart
        mopt["query_interval"] = query_interval
        mopt["refocus_ceil"] = refocus_ceil
        mopt["refocus_rebump"] = refocus_rebump
        mopt["refocus_reluctant"] = refocus_reluctant
        mopt["refocus_glue_sucks"] = glue_sucks
        mopt["refocus_glue_sucks_margin"] = glue_sucks_margin
        mopt["refocus_init_time"] = refocus_init_time
        mopt["subsume_rel_eff"] = subsume_rel_eff
        mopt["elim_rel_eff"] = elim_rel_eff
        mopt["rephase"] = bool(rephase)
        mopt["stabilize_only"] = bool(stabilize_only)
        mopt["walk"] = bool(walk)
        # mopt["gpu"] = False
        mopt["cpu_lim"] = timeout
        mopt["seed"] = seed
        mopt["config"] = config
        mopt_drat = deepcopy(mopt)
        mopt_drat["model"] = model_drat_path
        TMPOPTIONS.append(mopt_drat)

      OPTIONS = TMPOPTIONS

      if mem_per_worker is not None:
        worker_kwargs = {"num_cpus":1, "num_gpus":0, "memory":(mem_per_worker * 1024 * 1024 * 1024)}
      else:
        worker_kwargs = {"num_cpus":1, "num_gpus":0}

      workers = [ray.remote(**worker_kwargs)(DummyWorker).remote() for _ in range(n_workers)]
      for worker in workers:
        pin_to_core(worker)
      pool = ActorPool(workers)

      for k, w in enumerate(workers):
        ray.get(w.run_fn.remote((lambda: print(f"worker {k} online"))))

      for i, mopts in enumerate(OPTIONS):
        config_string = str(i)
        benchmark_name = Path(os.path.dirname(os.path.dirname(cnfdir))).stem +  "-" + Path(cnfdir).stem

        shopts = {"program_name":"cadical",
                  "prog_alias":(f"n-cdl-{i}"),
                  "benchmark":benchmark_name,
                  "prog_args":""}
        solver_name = "cadical"
        OUT_NAME =  f"{solver_name}-{name}-run-{i}.json" if not is_baseline(mopts) else f"{solver_name}-{name}-BASELINE-run-{i}.json"
        out_path = os.path.join(log_dir, benchmark_name, OUT_NAME)
        print("OUT PATH", out_path)
        deploy_GNN1_drat(model_cfg_path=cfg, ckpt_path=ckpt, save_path=model_drat_path)
        test_harness(cnfdir, mopts, shopts, out_path, pool=pool, exts=["cnf", "gz"], forbidden=["bz2", "xz"], solver=solver_name)
      for w in workers:
        del w
      del pool


if __name__ == "__main__":
  cfg_dict = vars(_parse_main())
  num_cpus = cfg_dict["n_workers"] + 1
  try:
    ray.init(address="auto", redis_password='5241590000000000')
  except:
    print("[WARNING] FALLING BACK ON SINGLE MACHINE RAY CLUSTER")
    ray.init()
  try:
    _main(**cfg_dict)
  except Exception as e:
    print("CAUGHT EXCEPTION IN MAIN")
    print(e)
    pass
