import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tempfile
import json
import datetime
import time
import torch.multiprocessing as mp

from util import check_make_path, files_with_extension
from gnn import *
from batch import Batcher
from data_util import H5Dataset, BatchedIterable, mk_H5DataLoader
from gen_data import NMSDP

def compute_softmax_kldiv_loss(logitss, probss):
  """
  Args:
    logitss: a list of 1D tensors
    probss: a list of 1D tensors, each of which is a valid probability distribution (entries sum to 1)

  Returns:
    averaged KL divergence loss
  """
  result = 0
  # loss = nn.KLDivLoss(reduction="sum")
  for logits, probs in zip(logitss, probss):
    probs = probs.squeeze() # required because of DataLoader magic
    # print("LOGITS", logits)
    # print("PROBS", probs)
    # print(logits.size())
    logits = F.log_softmax(logits, dim=0) # must be log-probabilities
    cl = F.kl_div(input=logits.view([1, logits.size(0)]), target=probs.view([1, probs.size(0)]), reduction="sum")
    # print("CL", cl)
    result += cl
    # result += F.kl_div(logits, probs, reduction="none")
  result = result / float(len(probss))
  return result

def compute_mask_loss(V_logitss, masks):
  """
  Computes softmax KL-divergence loss with respect to target uniform distributions represented by multi-hot masks
  """
  targets = [(x.float() * 1/x.sum()).view([x.size(0)]) for x in masks]
  # print("MASK LOSS TARGET EXAMPLE", targets[0])
  return compute_softmax_kldiv_loss(V_logitss, targets)

def compute_softmax_kldiv_loss_from_logits(V_logitss, target_logits, tau=4.0):
  softmax = nn.Softmax(dim=1)
  # print(target_logits[0])
  target_probs = []
  for logits in target_logits:
    logits = (logits - logits.mean())
    logits = logits/(logits.std() + 1e-10)
    logits = tau * logits
    target_probs.append(softmax((logits.view((1,) + logits.size()))).view(logits.size()))

  # target_probs = [softmax(tau * (logits.view((1,) + logits.size()))).view(logits.size()) for logits in target_logits]
  # # print("TARGET PROBS", target_probs[0])
  return compute_softmax_kldiv_loss(V_logitss, target_probs)

def NMSDP_to_sparse(nmsdp):
  C_idxs = torch.from_numpy(nmsdp.C_idxs)
  L_idxs = torch.from_numpy(nmsdp.L_idxs)
  indices = torch.stack([C_idxs.type(torch.long), L_idxs.type(torch.long)])
  values = torch.ones(len(C_idxs), device=indices.device)
  size = [nmsdp.n_clauses[0], 2*nmsdp.n_vars[0]]
  return torch.sparse.FloatTensor(indices=indices, values=values, size=size)

def NMSDP_to_sparse2(nmsdp): # needed because of some magic tensor coercion done by td.DataLoader constructor
  C_idxs = nmsdp.C_idxs[0]
  L_idxs = nmsdp.L_idxs[0]
  indices = torch.stack([C_idxs.type(torch.long), L_idxs.type(torch.long)])
  values = torch.ones(len(C_idxs), device=indices.device)
  size = [nmsdp.n_clauses[0], 2*nmsdp.n_vars[0]]
  return torch.sparse.FloatTensor(indices=indices, values=values, size=size)

def train_step(model, batcher, optim, nmsdps, device="cpu", CUDA_FLAG=False, use_NMSDP_to_sparse2=False, use_glue_counts=False):
  # the flag use_NMSDP_to_sparse2 should be True when we use mk_H5DataLoader instead of iterating over the H5Dataset directly, because DataLoader does magic conversions from numpy arrays to torch tensors
  optim.zero_grad()
  Gs = []
  var_lemma_countss = []
  core_var_masks = []
  core_clause_masks = []
  glue_countss = []

  def maybe_non_blocking(tsr):
    if CUDA_FLAG:
      return tsr.cuda(non_blocking=True)
    else:
      return tsr

  # print("USE GLUE COUNTS: ", use_glue_counts)
  for nmsdp in nmsdps:
    if not use_glue_counts:
      if not use_NMSDP_to_sparse2:
        Gs.append(maybe_non_blocking(NMSDP_to_sparse(nmsdp)))
        var_lemma_countss.append(maybe_non_blocking(torch.from_numpy(nmsdp.var_lemma_counts).type(torch.float32).squeeze()).to(device))
        core_var_masks.append(maybe_non_blocking(torch.from_numpy(nmsdp.core_var_mask).type(torch.bool).squeeze()).to(device))
        core_clause_masks.append(maybe_non_blocking(torch.from_numpy(nmsdp.core_clause_mask).type(torch.bool).squeeze()).to(device))
      else:
        Gs.append(maybe_non_blocking(NMSDP_to_sparse2(nmsdp)))
        var_lemma_countss.append(maybe_non_blocking(nmsdp.var_lemma_counts.type(torch.float32)[0]).to(device))
        core_var_masks.append(maybe_non_blocking(nmsdp.core_var_mask.type(torch.bool)[0]).to(device))
        core_clause_masks.append(maybe_non_blocking(nmsdp.core_clause_mask.type(torch.bool)[0]).to(device))
    else:
      if not use_NMSDP_to_sparse2:
        Gs.append(maybe_non_blocking(NMSDP_to_sparse(nmsdp)))
        glue_countss.append(maybe_non_blocking(torch.from_numpy(nmsdp.glue_counts).type(torch.float32).squeeze()).to(device))
      else:
        Gs.append(maybe_non_blocking(NMSDP_to_sparse2(nmsdp)))
        glue_countss.append(maybe_non_blocking(nmsdp.glue_counts.type(torch.float32)[0]).to(device))

  G = batcher(Gs)
  G.to(device)
  batched_V_drat_logits, batched_V_core_logits, batched_C_core_logits = (lambda x: (x[0].view([x[0].size(0)]), x[1].view([x[1].size(0)]), x[2].view([x[2].size(0)])))(model(G))
  # batched_V_drat_logits, batched_V_core_logits = (model(G))

  V_drat_logitss = batcher.unbatch(batched_V_drat_logits, mode="variable")
  if not use_glue_counts:
    V_core_logitss = batcher.unbatch(batched_V_core_logits, mode="variable")
    C_core_logitss = batcher.unbatch(batched_C_core_logits, mode="clause")

  # print("UNBATCHED DRAT LOGITS", [x.shape for x in V_drat_logitss])
  # print("DRAT LABELS", [x.shape for x in var_lemma_countss])

  # print("ok")

  # breakpoint()

  if use_glue_counts:
    drat_loss = compute_softmax_kldiv_loss_from_logits(V_drat_logitss, glue_countss, tau=1.0)
    core_loss = 0
    core_clause_loss = 0
  else:
    drat_loss = compute_softmax_kldiv_loss_from_logits(V_drat_logitss, var_lemma_countss, tau=1.0)
    core_loss = compute_mask_loss(V_core_logitss, core_var_masks)
    core_clause_loss = compute_mask_loss(C_core_logitss, core_clause_masks)
  # core_loss = 0
  # core_clause_loss = 0

  l2_loss = 0.0

  for param in model.parameters():
    l2_loss += (param**2).sum()
  # core_loss = 0
  # core_clause_loss = 0

  l2_loss = l2_loss * 1e-9

  # print("EXAMPLE CORE CLAUSE MASK", core_clause_masks[0])
  # print("EXAMPLE DRAT VAR COUNT", var_lemma_countss[0])

  loss = drat_loss + 0.1 * core_loss + 0.01 * core_clause_loss + l2_loss
  # loss = drat_loss
  loss.backward()

  nn.utils.clip_grad_value_(model.parameters(), 100)
  nn.utils.clip_grad_norm_(model.parameters(), 10)

  x = 0
  for name, param in model.named_parameters():
    # print(name, param.grad)
    try:
      g = param.grad
      x += g.norm()

      num_g_entries = torch.prod(torch.tensor(list(g.size())), 0)
      num_g_nonzero_entries = torch.nonzero(g).size(0)
      if not num_g_entries  == num_g_nonzero_entries:
        print( "G SIZE", num_g_entries, "LEN NONZEROS", num_g_nonzero_entries,  "OH NO ZERO GRAD AT", name, g, "AHHHHHHHH")
    except AttributeError:
      pass

  # for k, v in optim.state_dict().items():
  #   print(f"OPTIM KEY: {k} || OPTIM VALUE: {v}")

  optim.step()

  return drat_loss, core_loss, core_clause_loss, loss, x, l2_loss

def train_step2(model, optim, nmsdps, device="cpu", CUDA_FLAG=False, use_NMSDP_to_sparse2=False):
  optim.zero_grad()
  Gs = []
  core_var_masks = []
  var_lemma_countss = []

  def maybe_non_blocking(tsr):
    if CUDA_FLAG:
      return tsr.cuda(non_blocking=True)
    else:
      return tsr

  for nmsdp in nmsdps:
    if not use_NMSDP_to_sparse2:
      G = NMSDP_to_sparse(nmsdp)
      Gs.append(maybe_non_blocking(G))
      core_var_masks.append(maybe_non_blocking(torch.from_numpy(nmsdp.core_var_mask).type(torch.bool).squeeze()).to(device))
      var_lemma_countss.append(maybe_non_blocking(torch.from_numpy(nmsdp.var_lemma_counts).type(torch.float32).squeeze()).to(device))
    else:
      G = NMSDP_to_sparse2(nmsdp)
      Gs.append(maybe_non_blocking(G))
      core_var_masks.append(maybe_non_blocking(nmsdp.core_var_mask.type(torch.bool).squeeze()).to(device))
      var_lemma_countss.append(maybe_non_blocking(nmsdp.var_lemma_counts.type(torch.float32).squeeze()).to(device))

  V_drat_logitss, V_core_logitss = model(Gs)

  drat_loss = compute_softmax_kldiv_loss_from_logits(V_drat_logitss, var_lemma_countss)
  core_loss = compute_mask_loss(V_core_logitss, core_var_masks)

  loss = core_loss + drat_loss
  loss.backward()
  optim.step()

  return drat_loss, core_loss, loss


class TrainLogger:
  def __init__(self, logdir):
    self.logdir = logdir
    self.logfile = os.path.join(self.logdir, "log.txt")
    self.writer = SummaryWriter(log_dir=self.logdir)
    check_make_path(self.logdir)

  def write_scalar(self, name, value, global_step):
    self.writer.add_scalar(name, value, global_step)

  def write_log(self, *args):
    with open(self.logfile, "a") as f:
      print(f"{datetime.datetime.now()}:", *args)
      print(f"{datetime.datetime.now()}:", *args, file=f)

class Trainer:
  """
  v1 Trainer object. Only works for a single device.

  Args:
    model: nn.Module object
    dataset: iterable which yields batches of data
    optimizer: pytorch optimizer object
    ckpt dir: path to checkpoint directory
    ckpt_freq: number of gradient updates (i.e. batches) between saving checkpoints

  The `logger` attribute is a TrainLogger object, responsible for writing to training logs _and_ TensorBoard summaries.
  """
  def __init__(self, model, dataset, lr, ckpt_dir, ckpt_freq, restore=False, n_steps=-1, n_epochs=-1, index=0):
    self.model = model
    self.dataset = dataset
    self.ckpt_dir = ckpt_dir
    self.logger = TrainLogger(os.path.join(self.ckpt_dir, "logs/"))
    self.ckpt_freq = ckpt_freq
    self.save_counter = 0
    self.GLOBAL_STEP_COUNT = 0
    self.n_steps = None if n_steps == -1 else n_steps
    self.n_epochs = 1 if n_epochs == -1 else n_epochs
    self.CUDA_FLAG = torch.cuda.is_available()
    # self.device = torch.device(f"cuda:{index}" if self.CUDA_FLAG else "cpu")
    self.device = torch.device("cuda" if self.CUDA_FLAG else "cpu")
    print("CUDA FLAG", self.CUDA_FLAG)
    self.model.to(self.device)
    self.lr = lr
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98))
    util.check_make_path(self.ckpt_dir)
    if restore:
      try:
        self.load_latest_ckpt()
      except IndexError:
        pass

  def save_model(self, model, optimizer, ckpt_path): # TODO(jesse): implement a CheckpointManager
    torch.save({
      "model_state_dict":model.state_dict(),
      "optimizer_state_dict":optimizer.state_dict(),
      "save_counter":self.save_counter,
      "GLOBAL_STEP_COUNT":self.GLOBAL_STEP_COUNT
    }, ckpt_path)

  def load_model(self, model, optimizer, ckpt_path):
    """
    Loads `model` and `optimizer` state from checkpoint at `ckpt_path`

    Args:
      model: nn.Module object
      optimizer: PyTorch optimizer object
      ckpt_path: path to checkpoint containing serialized model state and optimizer state

    Returns:
      Nothing.
    """
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    self.save_counter = ckpt["save_counter"]
    self.GLOBAL_STEP_COUNT = ckpt["GLOBAL_STEP_COUNT"]
    print(f"LOADED MODEL, SAVE COUNTER: {self.save_counter}, GLOBAL_STEP_COUNT = {self.GLOBAL_STEP_COUNT}")

  def get_latest_from_index(self, ckpt_dir):
    """
    Args:
      ckpt_dir: checkpoint directory

    Returns:
      a dict cfg_dict such that cfg_dict["latest"] is the path to the latest checkpoint
    """
    index = files_with_extension(ckpt_dir, "index")[0]
    with open(index, "r") as f:
      cfg_dict = json.load(f)
    return cfg_dict["latest"]

  def update_index(self, ckpt_path):
    """
    Dump a JSON to a `.index` file, pointing to the most recent checkpoint.
    """
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

  def maybe_save_ckpt(self, GLOBAL_STEP_COUNT, force_save=False):
    """
    Saves the model if GLOBAL_STEP_COUNT is at the ckpt_freq.

    Returns a bit indicating whether or not the model was saved.
    """
    if (int(GLOBAL_STEP_COUNT) % self.ckpt_freq == 0) or force_save:
      ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{self.save_counter}.pth")
      self.save_model(self.model, self.optimizer, ckpt_path)
      self.update_index(ckpt_path)
      self.logger.write_log(f"[TRAIN LOOP] Wrote checkpoint to {ckpt_path}.")
      self.save_counter += 1
      return True
    return False

  def load_latest_ckpt(self):
    ckpt_path = self.get_latest_from_index(self.ckpt_dir)
    self.load_model(self.model, self.optimizer, ckpt_path)
    self.logger.write_log(f"[TRAIN LOOP] Loaded weights from {ckpt_path}.")

  def train(self):
    self.logger.write_log(f"[TRAIN LOOP] HYPERPARAMETERS: LR {self.lr}")
    self.logger.write_log(f"[TRAIN LOOP] NUM_EPOCHS: {self.n_epochs}")
    check_make_path(self.ckpt_dir)
    self.logger.write_log("[TRAIN LOOP] Starting training.")
    batcher = Batcher(device=self.device)
    saved = False
    for epoch_count in range(self.n_epochs):
      self.logger.write_log(f"[TRAIN LOOP] STARTING EPOCH {epoch_count}")
      for nmsdps in self.dataset:
        drat_loss, core_loss, core_clause_loss, loss, grad_norm, l2_loss = train_step(self.model, batcher, self.optimizer, nmsdps, device=self.device, CUDA_FLAG=self.CUDA_FLAG, use_NMSDP_to_sparse2=True)

        self.logger.write_scalar("drat_loss", drat_loss, self.GLOBAL_STEP_COUNT)
        self.logger.write_scalar("core_loss", core_loss, self.GLOBAL_STEP_COUNT)
        self.logger.write_scalar("core_clause_loss", core_clause_loss, self.GLOBAL_STEP_COUNT)
        self.logger.write_scalar("total_loss", loss, self.GLOBAL_STEP_COUNT)
        self.logger.write_scalar("LAST GRAD NORM", grad_norm, self.GLOBAL_STEP_COUNT)
        self.logger.write_scalar("l2_loss", l2_loss, self.GLOBAL_STEP_COUNT)
        self.logger.write_log(f"[TRAIN LOOP] Finished global step {self.GLOBAL_STEP_COUNT}. Loss: {loss}.")
        self.GLOBAL_STEP_COUNT += 1
        saved = self.maybe_save_ckpt(self.GLOBAL_STEP_COUNT)
        if self.n_steps is not None:
          if self.GLOBAL_STEP_COUNT >= self.n_steps:
            break
    if not saved:
      self.maybe_save_ckpt(self.GLOBAL_STEP_COUNT, force_save=True) # save at the end of every epoch regardless

def gen_nmsdp_batch(k):
  nmsdps = []
  D = RandKCNFDataset(3,40)
  D_gen = D.__iter__()
  for _ in range(k):
    cnf = next(D_gen)
    with tempfile.TemporaryDirectory() as tmpdir:
      nmsdp = gen_nmsdp(tmpdir, cnf, is_train=True, logger=DummyLogger())
    nmsdps.append(nmsdp)
  return nmsdps

def _parse_main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--cfg", type=str, dest="cfg", action="store")
  parser.add_argument("--lr", type=float, dest="lr", action="store")
  parser.add_argument("--data-dir", type=str, dest="data_dir", action="store")
  parser.add_argument("--batch-size", type=int, dest="batch_size", action="store")
  parser.add_argument("--n-data-workers", type=int, dest="n_data_workers", action="store")
  parser.add_argument("--ckpt-dir", type=str, dest="ckpt_dir", action="store")
  parser.add_argument("--ckpt-freq", type=int, dest="ckpt_freq", action="store")
  parser.add_argument("--n-steps", type=int, dest="n_steps", action="store", default=-1)
  parser.add_argument("--n-epochs", type=int, dest="n_epochs", action="store", default=-1)
  parser.add_argument("--forever", action="store_true")
  parser.add_argument("--index", action="store", default=0, type=int)
  opts = parser.parse_args()

  return opts

def _main_train1(cfg=None, opts=None):
  if opts is None:
    opts = _parse_main()

  if cfg is None:
    # cfg = defaultGNN1Cfg
    with open(opts.cfg, "r") as f:
      cfg = json.load(f)

  model = GNN1(**cfg)

  dataset = mk_H5DataLoader(opts.data_dir, opts.batch_size, opts.n_data_workers)
  trainer = Trainer(model, dataset, opts.lr, ckpt_dir=opts.ckpt_dir, ckpt_freq=opts.ckpt_freq, restore=True, n_steps=opts.n_steps, n_epochs=opts.n_epochs, index=opts.index)

  if opts.forever is True:
    while True:
      trainer.train()
  else:
    trainer.train()

def _test_trainer():
  model = GNN1(**defaultGNN1Cfg)
  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  dataset = mk_H5DataLoader("./train_data/", batch_size=16, num_workers=2)
  trainer = Trainer(model, dataset, optimizer, ckpt_dir="./test_weights/", ckpt_freq=10, restore=True, n_steps=opts.n_steps)
  for _ in range(5):
    trainer.train()
    # trainer.load_latest_ckpt()

GNN1Cfg0 = {
  "clause_dim":64,
  "lit_dim":16,
  "n_hops":1,
  "n_layers_C_update":0,
  "n_layers_L_update":0,
  "n_layers_score":1,
  "activation":"leaky_relu"
}
