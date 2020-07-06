"""
Distributed training of GNN1.
"""

import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
from batch import Batcher
import json

import ray
from ray.util.sgd.torch.training_operator import TrainingOperator

import warnings

from ray.util.sgd.utils import TimerStat, AverageMeter
from ray.util.sgd.torch.constants import (
    SCHEDULER_STEP_EPOCH, SCHEDULER_STEP_BATCH, SCHEDULER_STEP, BATCH_COUNT)

from sgd import TorchTrainer
from train1 import train_step, TrainLogger
from gnn import *
from data_util import *
import util

DTrainCfg = collections.namedtuple(
  "DTrainCfg",
  [
    "logdir",
    "modelcfg", # model configuration dict
    "optimizer",
    "lr",
    "data_dir",
    "ckpt_dir",
    "ckpt_freq",
    "batch_size",
    "t0"
  ]
)

testcfg = DTrainCfg(
  logdir = "./dtrain/logs/",
  modelcfg = defaultGNN1Cfg,
  optimizer = "adam",
  lr = 1e-4,
  data_dir = "./train_data/",
  ckpt_dir = "./dtrain/weights/",
  ckpt_freq = 10,
  batch_size = 8,
  t0 = 1e6
)

def model_creator(config):
  return GNN1(**config["modelcfg"])

def optimizer_creator(model, config):
  optim_name = config["optimizer"]

  if optim_name == "adam":
    return optim.Adam(model.parameters(), lr=config["lr"])
  elif optim_name == "adamw":
    return optim.AdamW(model.parameters(), lr=config["lr"])
  elif optim_name == "asgd":
    return optim.ASGD(model.parameters(), lr=config["lr"], t0=config["t0"])
  elif optim_name == "sgd":
    return optim.SGD(model.parameters(), lr=config["lr"])  
  else:
    raise Exception(f"unsupported optimizer {optim_name}")

def data_creator(config):
    return H5Dataset(config["data_dir"], config["batch_size"])

def loss_creator(config): # this is just a placeholder, the actual loss function is hardcoded in train_fn
    return torch.nn.MSELoss()

DATALOADER_CONFIG = {
  "worker_init_fn":h5_worker_init_fn
}

def update_index(ckpt_path):
  ckpt_dir = os.path.dirname(ckpt_path)
  index_files = files_with_extension(ckpt_dir, "index")
  if len(index_files) == 0:
    index = os.path.join(ckpt_dir, "latest.index")
  else:
    assert len(index_files) == 1
    index = index_files[0]
  with open(index, "w") as f:
    cfg_dict = {"latest":ckpt_path}
    f.write(json.dumps(cfg_dict, indent=2))

def get_latest_ckpt(ckpt_dir):
  index = files_with_extension(ckpt_dir, "index")[0]
  with open(index, "r") as f:
    cfg_dict = json.load(f)
  return cfg_dict["latest"]

class GNN1TrainingOperator(TrainingOperator):
  def __init__(self, *args, **kwargs):
    try:
      self.ckpt_dir = kwargs.pop("ckpt_dir")
      self.ckpt_freq = kwargs.pop("ckpt_freq")
      self.index =  kwargs.pop("index")
      self.save_counter = 0
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.lr = kwargs.pop("lr")
    except KeyError:
      raise Exception("DING DONG!")

    super(GNN1TrainingOperator, self).__init__(*args, **kwargs)
    self.batcher = Batcher(device=self.device)
    if self.index == 0:
      self.logger = TrainLogger(logdir=args[0].get("logdir"))
      self.logger.write_log("SOME HYPERPARAMETERS:")
      for key, value in self.config.items():
        if key == "lr" or key == "optimizer" or key == "batch_size" or key == "ckpt_freq" or key == "data_dir":
          self.logger.write_log(f"{key}: {value}") # oops
    self.optim = self.optimizers[0]
    try:
      latest = get_latest_ckpt(self.ckpt_dir)
      print(f"LOADING WEIGHTS FROM {latest}")
      state = torch.load(latest)
      self.model.load_state_dict(state["models"][0], strict=False)
      self.optim.load_state_dict(state["optimizers"][0])
      if self.lr is not None:
        print(f"FORCING LEARNING RATE: {self.lr}")
        for g in self.optim.param_groups:
          g["lr"] = self.lr
      self.save_counter = state["SAVE_COUNTER"]
      self.global_step = state["GLOBAL_STEP"]
      print(f"SUCCESSFULLY LOADED CHECKPOINT FROM {latest}")
      self.save_counter += 1
    except Exception as e:
      print("WARNING: EXCEPTION ", e)
      pass


  def save(self):
    if self.index == 0:
      ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{self.save_counter}.pth")
      torch.save(
        {
          "models":[{k:v.cpu() for k,v in self.model.state_dict().items()}],
          "optimizers":[self.optim.state_dict()],
          "GLOBAL_STEP":self.global_step,
          "SAVE_COUNTER":self.save_counter
        }, ckpt_path)

      update_index(ckpt_path)
      self.logger.write_log(f"[TRAIN LOOP -- HEAD WORKER] SAVED CHECKPOINT TO {ckpt_path}")

  def train_epoch(self, iterator, info):
      self._losses = AverageMeter()

      self.model.train()
      try:
        with self.timers["epoch_time"]:
            for batch_idx, batch in enumerate(iterator):
                batch_info = {
                    "batch_idx": batch_idx,
                    "global_step": self.global_step
                }
                batch_info.update(info)
                metrics = self.train_batch(batch, batch_info=batch_info)

                if self.scheduler and batch_info.get(
                        SCHEDULER_STEP) == SCHEDULER_STEP_BATCH:
                    self.scheduler.step()

                if "loss" in metrics:
                    self._losses.update(
                        metrics["loss"], n=metrics.get("num_samples", 1))
                self.global_step += 1
      except Exception as e:
        print("CAUGHT EXCEPTION", e)
      finally:
        if self.index == 0:
          print("[TRAINER] FINALIZING EPOCH AND SAVING")
          self.save() # save no matter what

      if self.scheduler and info.get(SCHEDULER_STEP) == SCHEDULER_STEP_EPOCH:
          self.scheduler.step()

      stats = {
          BATCH_COUNT: batch_idx + 1,
          "mean_train_loss": self._losses.avg,
          "last_train_loss": self._losses.val,
          "epoch_time": self.timers["epoch_time"].last
      }
      stats.update({
          timer_tag: timer.mean
          for timer_tag, timer in self.timers.items()
      })
      return stats


  def train_batch(self, batch, batch_info):
    num_samples = len(batch)
    drat_loss, core_loss, core_clause_loss, loss, x, l2_loss = train_step(self.model,
                                                                          self.batcher,
                                                                          self.optim,
                                                                          batch,
                                                                          use_NMSDP_to_sparse2=True,
                                                                          CUDA_FLAG=torch.cuda.is_available(),
                                                                          device=self.device,
                                                                          use_glue_counts=True)
    GLOBAL_STEP_COUNT = batch_info.get("global_step")

    if self.index == 0:
      self.logger.write_log(f"[TRAIN LOOP] Finished global step {GLOBAL_STEP_COUNT}. Loss: {loss}.")
      self.logger.write_scalar("drat_loss", drat_loss, GLOBAL_STEP_COUNT)
      self.logger.write_scalar("core_loss", core_loss, GLOBAL_STEP_COUNT)
      self.logger.write_scalar("core_clause_loss", core_clause_loss, GLOBAL_STEP_COUNT)
      self.logger.write_scalar("loss", loss, GLOBAL_STEP_COUNT)
      if (GLOBAL_STEP_COUNT + 1) % self.ckpt_freq == 0:
        if self.save_counter == 0:
          util.check_make_path(self.ckpt_dir)
        self.save()
        self.save_counter += 1

    return {
      "drat_loss":drat_loss,
      "core_loss":core_loss,
      "loss":loss,
      "num_samples":num_samples
    }

def parse_config():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", dest="logdir", action="store")
  parser.add_argument("--modelcfg", dest="modelcfg", action="store", default=None)
  parser.add_argument("--optimizer", dest="optimizer", action="store", default="adam")
  parser.add_argument("--lr", dest="lr", action="store", type=float)
  parser.add_argument("--data-dir", dest="data_dir", action="store")
  parser.add_argument("--ckpt-dir", dest="ckpt_dir", action="store")
  parser.add_argument("--ckpt-freq", dest="ckpt_freq", action="store", type=int)
  parser.add_argument("--batch-size", dest="batch_size", action="store", type=int)
  parser.add_argument("--num-replicas", dest="num_replicas", action="store", type=int)
  parser.add_argument("--use-gpu", dest="use_gpu", action="store_true")
  parser.add_argument("--n-data-workers", dest="n_data_workers", action="store", type=int, default=0)
  parser.add_argument("--n-epochs", dest="num_epochs", action="store",type=int, default=1)
  parser.add_argument("--t0", dest="t0", action="store",type=float,default=1e6)
  opts = parser.parse_args()
  if opts.modelcfg is None:
    opts.modelcfg = defaultGNN1Cfg
  else:
    with open(opts.modelcfg, "r") as f:
      cfg_dict = json.load(f)
    opts.modelcfg = cfg_dict
  return opts

def parallel_train1(num_replicas, cfg, n_data_workers=0, num_epochs=1):
  DATALOADER_CONFIG["num_workers"] = n_data_workers
  try:
    ray.init(address="auto", redis_password='5241590000000000')
  except:
    print("[WARNING] FALLING BACK ON SINGLE MACHINE CLUSTER")
    ray.init()
  trainer = TorchTrainer(
    model_creator,
    data_creator,
    optimizer_creator,
    loss_creator=nn.MSELoss,
    config=cfg,
    ckpt_dir = cfg["ckpt_dir"],
    ckpt_freq = cfg["ckpt_freq"],
    dataloader_config = DATALOADER_CONFIG,
    num_replicas=num_replicas,
    batch_size=num_replicas, # move batching into the datasets
    use_gpu=cfg["use_gpu"],
    training_operator_cls=GNN1TrainingOperator
  )

  for _ in range(num_epochs):
    trainer.train()

  ray.shutdown()

def _main():
  cfg = vars(parse_config())
  num_replicas = cfg.pop("num_replicas")
  n_data_workers = cfg.pop("n_data_workers")
  num_epochs = cfg.pop("num_epochs")
  return parallel_train1(num_replicas, cfg, n_data_workers, num_epochs)

if __name__ == "__main__":
  _main()
