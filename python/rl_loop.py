import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.signal
import time
import os
import json
import ray
from ray.util import ActorPool
import random
import queue
from queue import Queue
import datetime
from copy import deepcopy

from satenv import SatEnv
from gnn import defaultGNN1Cfg
from gnn import rl_GNN1 as GNN1
from batch import Batcher
from util import files_with_extension, recursively_get_files
import threading


# TODO: the abstractions in this file suck

def check_zero_grads(model):
  x = 0
  for name, param in model.named_parameters():
    try:
      g = param.grad
      x += g.norm()

      num_g_entries = torch.prod(torch.tensor(list(g.size())), 0)
      num_g_nonzero_entries = torch.nonzero(g).size(0)
      if not num_g_nonzero_entries == num_g_entries:
        print( "G SIZE", num_g_entries, "LEN NONZEROS", num_g_nonzero_entries,  "OH NO ZERO GRAD AT", name, g, "HOLA BUENOS DIAS")

      num_g_nan_entries = torch.isnan(g).to(dtype=torch.int32).sum()
      if not int(num_g_nan_entries) == 0:
        print("NUM G NAN ENTRIES: ", int(num_g_nan_entries))
        print( "G SIZE", num_g_entries, "OH NO NAN GRAD AT", name, g, "HOLA BUENOS DIAS")
    except AttributeError:
      pass
  print("GRAD NORMS: ", x)

def discount_cumsum(x, discount=1):
  """
  magic from rllab for computing discounted cumulative sums of vectors.
  input:
  vector x,
    [x0,
     x1,
     x2]
  output,
    [x0 + discount * x1 + discount^2 * x2,
     x1 + discount * x2,
     x2]
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def mk_G(CL_idxs):
  C_idxs = np.array(CL_idxs.C_idxs, dtype="int32")
  L_idxs = np.array(CL_idxs.L_idxs, dtype="int32")
  indices = torch.stack([torch.as_tensor(C_idxs).to(torch.long), torch.as_tensor(L_idxs).to(torch.long)])
  values = torch.ones(len(C_idxs), device=indices.device)
  size = [CL_idxs.n_clauses, 2*CL_idxs.n_vars]
  return torch.sparse.FloatTensor(indices=indices, values=values, size=size)

def softmax_sample_from_logits(logits, exploration_eps=0.01):
  explore = np.random.choice([True,False], p=[exploration_eps, 1.0-exploration_eps])
  try:
    if not explore:
      dist = Categorical(logits=logits)
    else:
      dist = Categorical(logits=torch.zeros(*logits.size(), dtype=torch.float32))
  except ValueError:
    print("BAD LOGITS: ", str(logits))
    return np.random.choice(range(len(logits)))
  try:
    return dist.sample()
  except RuntimeError:
    print("BAD LOGITS: ", logits)
    return np.random.choice(range(len(logits)))


def sample_trajectory(agent, env, profile=False):
  """
  Samples a trajectory from the environment and then resets it. This assumes the environment has been initialized.
  """
  Gs = []
  mu_logitss = []
  actions = []
  rewards = []
  value_estimates = []

  CL_idxs = env.render()
  terminal_flag = False
  while (not terminal_flag):
    G = mk_G(CL_idxs)
    if profile:
      start = time.time()
    mu_logits, value_estimate = agent.act(G)
    if profile:
      elapsed = time.time() - start
      print(f"[sample_trajectory] inference took {elapsed}s")
    action = (softmax_sample_from_logits(mu_logits, exploration_eps=0.02)+1) # torch multinomial zero-indexes
    if profile:
      start = time.time()
    env_result = env.step((np.random.choice([1,-1]))*action)
    if profile:
      elapsed = time.time() - start
      print(f"[sample_trajectory] env step took {elapsed}s")
    CL_idxs, reward, terminal_flag = env_result.obs, env_result.reward, env_result.is_terminal
    Gs.append(G)
    mu_logitss.append(mu_logits)
    actions.append(action)
    rewards.append(reward)
    value_estimates.append(value_estimate)

  env.reset()
  return Gs, mu_logitss, actions, rewards, value_estimates

def process_trajectory(Gs, mu_logitss, actions, rewards, vals, last_val=0, gam=1.0, lam=1.0):
  rewards = np.array(rewards)
  gs = discount_cumsum(rewards, gam)
  deltas = np.append(rewards, last_val)[:-1] + gam * np.append(vals, last_val)[1:] - np.append(vals, last_val)[:-1]
  adv = discount_cumsum(deltas, gam*lam)
  episode_lengths = [len(rewards) - i for i in range(len(rewards))]
  result = Gs, mu_logitss, actions, gs, adv, episode_lengths
  result_aux = list(zip(*result))
  random.shuffle(result_aux)
  return result_aux, (gs[0], len(result_aux))   # note, gs[0] is total value

class Agent:
  def __init__(self):
    pass

  def act(self, G): # returns action and value estimate; agent handles softmax-sampling
    raise NotImplementedError

class RandomAgent(Agent):
  def __init__(self, seed=None):
    if seed is not None:
      torch.manual_seed(seed)

  def act(self, G):
    n_vars = int(G.size()[1]/2)
    # return torch.rand(n_vars), 0
    return torch.zeros(n_vars, dtype=torch.float32), 0.0

class Z3Agent(Agent):
  def __init__(self):
    pass

  def act(self, G):
    indices = G.coalesce().indices()
    C_idxs, L_idxs = indices[0], indices[1]
    size = G.size()
    n_vars = int(size[1]/2)
    clauses = [[] for _ in range(size[0])]
    def decode_l_idx(k):
      if k + 1 > n_vars:
        return -(k+1 - n_vars)
      else:
        return k +1

    for c_idx, l_idx in zip(C_idxs, L_idxs):
      clauses[c_idx].append(int(decode_l_idx(l_idx)))
    fmla = CNF(from_clauses=clauses)
    action = Z3_select_var(fmla)
    result = torch.full((n_vars,), fill_value=-1000.0, dtype=torch.float32)
    result[action] = 1000.0
    return result, 0.0

class NeuroAgent(Agent):
  def __init__(self, model_cfg=defaultGNN1Cfg, model_state_dict=None):
    self.model = GNN1(**model_cfg)
    # self.model = rl_GNN1(**model_cfg)
    if model_state_dict is not None:
      self.model.load_state_dict(model_state_dict)

  def act(self, G):
    p_logits, v_pre_logits = self.model(G)
    # p_logits = self.model(G)
    return p_logits.view(p_logits.size(0)).detach(), torch.tanh(v_pre_logits.mean()).detach()

  def set_weights(self, model_state_dict):
    self.model.load_state_dict(model_state_dict)

class EpisodeWorker: # learner is a handle to a ReplayLearnerfer object
  def __init__(self, learner, weight_manager, model_cfg=defaultGNN1Cfg, model_state_dict=None, from_cnf=None, from_file=None, seed=None, sync_freq=10, restore=True):
    print("WORKER ONLINE")
    self.learner = learner
    self.weight_manager = weight_manager
    if seed is not None:
      torch.manual_seed(seed)
      np.random.seed(seed)
    if from_cnf is not None or from_file is not None:
      self.set_env(from_cnf, from_file)
    self.sync_freq = sync_freq
    self.ckpt_rank = 0
    self.trajectory_count = 0
    self.agent = NeuroAgent(model_cfg=model_cfg)
    if restore:
      self.try_update_weights()
    print("WORKER INITIALIZED")

  def set_env(self, from_cnf=None, from_file=None):
    print("SETTING ENV")
    if from_cnf is not None:
      self.td = tempfile.TemporaryDirectory()
      cnf_path = os.path.join(self.td.name, str(uuid.uuid4()) + ".cnf")
      from_cnf.to_file(cnf_path)
      try:
        self.env = SatEnv(cnf_path)
      except RuntimeError as e:
        print("BAD CNF:", cnf_path)
        raise e
    elif from_file is not None:
      try:
        self.env = SatEnv(from_file)
      except RuntimeError as e:
        print("BAD CNF:", from_file)
        raise e
    else:
      raise Exception("must set env with CNF or file")

  def sample_trajectory(self):
    tau = sample_trajectory(self.agent, self.env)
    ray.get(self.learner.ingest_trajectory.remote(process_trajectory(*tau)))
    # print(f"SAMPLED TRAJECTORY OF LENGTH {len(tau[0])}")
    self.trajectory_count += 1
    if self.trajectory_count % self.sync_freq:
      self.try_update_weights()

  def set_weights(self, model_state_dict, new_rank):
    self.agent.set_weights(model_state_dict)
    self.ckpt_rank = new_rank

  def try_update_weights(self):
    status, new_state_dict, new_rank = ray.get(self.weight_manager.sync_weights.remote(self.ckpt_rank))
    if status:
      print(f"SYNCING WEIGHTS: {self.ckpt_rank} -> {new_rank}")
      self.set_weights(new_state_dict, new_rank)

  def __del__(self):
    try:
      self.td.cleanup()
    except AttributeError:
      pass

class ReplayBufferWriter(threading.Thread):
  def __init__(self, queue, writer):
    threading.Thread.__init__(self)
    self.queue = queue
    self.writer = writer
    self.episode_count = 0

  def ingest_trajectory(self, tau):
    while True:
      if self.queue.qsize() > 10000:
        print("[ReplayBuffer] QUEUE FULL, SLEEPING")
        time.sleep(1.0)
      else:
        break

    result_aux, (total_return, episode_length) = tau
    self.writer.add_scalar("total return", total_return, self.episode_count)
    self.writer.add_scalar("episode length", episode_length, self.episode_count)
    for G, mu_logits, action, g, adv, episode_length in result_aux:
      self.queue.put((G, mu_logits, action, g, adv, episode_length))

    print(f"TOTAL RETURN: {total_return}")
    self.episode_count += 1

class ReplayBufferBatcher(threading.Thread):
  def __init__(self, queue, batch_queue, batch_size):
    threading.Thread.__init__(self)
    self.queue = queue
    self.batch_queue = batch_queue
    self.batch_size = batch_size

  def run(self):
    while True:
      if self.batch_size <= self.queue.qsize():
        batch = self.get_batch()
        self.batch_queue.put(batch)
      else:
        time.sleep(0.25)

  def get_batch(self):
    batch_size = self.batch_size
    Gs = []
    mu_logitss = []
    actions = []
    gs = []
    advs = []
    episode_lengths = []
    for _ in range(batch_size):
      G, mu_logits, action, g, adv, episode_length = self.queue.get()
      Gs.append(G)
      mu_logitss.append(mu_logits)
      actions.append(action)
      gs.append(g)
      advs.append(adv)
      episode_lengths.append(episode_length)
    return Gs, mu_logitss, actions, gs, advs, episode_lengths

class ReplayBufferReader(threading.Thread):
  def __init__(self, queue, batch_queue):
    threading.Thread.__init__(self)
    self.queue = queue
    self.batch_queue = batch_queue

  def get_batch(self, batch_size, sample_randomly=False): # WARNING: DUMMY ARGUMENTS
    return self.batch_queue.get()

  def get_frame(self):
    return self.queue.get()

class ReplayBuffer:
  def __init__(self, logdir, batch_size, limit=100000):
    print("BUFFER INITIALIZING")
    self.logdir = logdir
    self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, "returns/"))
    self.queue = Queue()
    self.batch_queue = Queue()
    self.sample_count = 0
    self.limit = limit
    self.reader = ReplayBufferReader(self.queue, self.batch_queue)
    self.batcher = ReplayBufferBatcher(self.queue, self.batch_queue, batch_size)
    self.writer_thread = ReplayBufferWriter(self.queue, self.writer)
    self.batcher.start()
    self.reader.start()
    self.writer_thread.start()
    self.batch_size = batch_size
    print("BUFFER INITIALIZED")

  def set_episode_count(self, episode_count):
    print("[ReplayBuffer] setting episode count: ", episode_count)
    self.writer_thread.episode_count = episode_count

  def get_episode_count(self):
    return self.writer_thread.episode_count

  def ingest_trajectory(self,tau):
    self.writer_thread.ingest_trajectory(tau)

  def batch_ready(self, batch_size):
    # return batch_size <= self.queue.qsize()
    return self.batch_queue.qsize() > 0

  def frame_ready(self):
    return self.queue.qsize() > 0

  def get_batch(self, batch_size, sample_randomly=False):
    return self.reader.get_batch(batch_size,sample_randomly)

  def get_frame(self):
    return self.reader.get_frame()

def train_step(model, optim, scheduler, batcher, Gs,
               mu_logitss, actions, gs,
               advs, episode_lengths,
               device=torch.device("cpu"), GLOBAL_STEP_COUNT=None):
  batch_size = len(Gs)
  G = batcher.batch(Gs)
  pre_policy_logitss, pre_unreduced_value_logitss = model(G)
  policy_logitss = batcher.unbatch(pre_policy_logitss, mode="variable")

  actions = torch.as_tensor(np.array(actions, dtype=np.int_)).to(device)
  advs = torch.as_tensor(np.array(advs, dtype="float32")).to(device)

  advs = advs - advs.mean()
  advs = advs / (torch.std(advs) + 1e-8)

  policy_distribs = [Categorical(logits=x.view(x.size(0)).to(device)*2.0) for x in policy_logitss]
  mu_distribs = [Categorical(logits=x.view(x.size(0)).to(device)*2.0) for x in mu_logitss]

  target_log_probs = []
  behavior_log_probs = []

  for i, (x,y, a) in enumerate(zip(policy_distribs, mu_distribs, actions)):
    target_log_probs.append(x.log_prob(a - 1))
    behavior_log_probs.append(y.log_prob(a - 1))

  target_log_probs = torch.tensor(target_log_probs).to(device)
  behavior_log_probs = torch.tensor(behavior_log_probs).to(device)

  log_rhos = target_log_probs - behavior_log_probs

  log_rhos = log_rhos.detach().view(*actions.size())
  rhos = torch.exp(log_rhos).squeeze()

  entropies = torch.tensor([x.entropy() for x in policy_distribs]).to(device)
  entropy_bonus = entropies.mean()

  psis = advs * rhos

  nll_loss = -target_log_probs
  nll_loss = (nll_loss * psis).mean()
  vals = torch.tanh(torch.stack([x.mean() for x in batcher.unbatch(pre_unreduced_value_logitss, mode="variable")]).to(device))
  v_loss = F.mse_loss(vals, torch.as_tensor(np.array(gs, dtype="float32")).to(device))
  reg_loss = torch.tensor(0.0).to(device)
  for w in model.parameters():
    reg_loss += torch.norm(w, 2)

  p_loss = nll_loss + -(0.05 * entropy_bonus)

  optim.zero_grad()
  loss = p_loss + 0.1 * v_loss + (1e-10) * reg_loss
  loss.backward()
  nn.utils.clip_grad_value_(model.parameters(), 150)
  nn.utils.clip_grad_norm_(model.parameters(), 15)
  optim.step()
  
  lr = torch.tensor(scheduler.get_last_lr(), dtype=torch.float32).mean().detach().cpu().numpy()

  return {"p_loss":p_loss.detach().cpu().numpy(),
          "entropy_bonus":entropy_bonus.detach().cpu().numpy(),
          "v_loss":v_loss.detach().cpu().numpy(),
          "learning_rate":lr}

class WeightManager:
  def __init__(self, ckpt_dir):
    print("WEIGHT MANAGER INITIALIZED")
    self.ckpt_dir = os.path.expandvars(ckpt_dir)
    self.save_counter = 0
    self.model_state_dict = dict()
    self.optim_state_dict = dict()
    self.scheduler_state_dict = dict()
    self.GLOBAL_STEP_COUNT = 0
    self.max_to_keep=25

  def get_latest_from_index(self, ckpt_dir):
    index = files_with_extension(ckpt_dir, "index")[0]
    with open(index, "r") as f:
      cfg_dict = json.load(f)

    self.save_counter = cfg_dict["save_counter"]
    return cfg_dict["latest"]

  def load_latest_ckpt(self):
    try:
      ckpt_path = self.get_latest_from_index(self.ckpt_dir)
    except IndexError:
      print("[WEIGHT MANAGER] NO INDEX FOUND")
      return None
    ckpt = torch.load(ckpt_path)
    self.load_ckpt(ckpt)
    print(f"[WEIGHT MANAGER] RESTORING FROM {ckpt_path}")
    return ckpt

  def load_ckpt_from_path(self, ckpt_path):
    return self.load_ckpt(torch.load(ckpt_path))

  def load_ckpt(self, ckpt):
    self.model_state_dict = ckpt["model_state_dict"]
    self.optim_state_dict = ckpt["optim_state_dict"]
    self.scheduler_state_dict = ckpt["scheduler_state_dict"]
    self.save_counter = ckpt["save_counter"]
    self.GLOBAL_STEP_COUNT = ckpt["GLOBAL_STEP_COUNT"]

  def sync_weights(self, rank):
    status = False
    model_state_dict = None
    new_rank = rank
    if rank < self.save_counter:
      status = True
      model_state_dict = dict()
      for k,v in self.model_state_dict.items():
        model_state_dict[k] = v.cpu()
      new_rank = self.save_counter
    return status, model_state_dict, new_rank

  def update_index(self, ckpt_path, save_counter):
    ckpt_dir = os.path.dirname(ckpt_path)
    index_files = files_with_extension(ckpt_dir, "index")
    if len(index_files) == 0:
      index = os.path.join(ckpt_dir, "latest.index")
    else:
      assert len(index_files) == 1
      index = index_files[0]
    with open(index, "w") as f:
      cfg_dict = {"latest":ckpt_path, "save_counter":save_counter}
      f.write(json.dumps(cfg_dict, indent=2))
    try:
      os.remove(os.path.join(ckpt_dir, f"ckpt_{save_counter-self.max_to_keep}.pth"))
    except:
      pass

  def save_ckpt(self, model_state_dict, optim_state_dict, scheduler_state_dict, save_counter, GLOBAL_STEP_COUNT, episode_count):
    self.model_state_dict = model_state_dict
    self.optim_state_dict = optim_state_dict
    self.scheduler_state_dict = scheduler_state_dict
    self.save_counter = save_counter
    self.GLOBAL_STEP_COUNT = GLOBAL_STEP_COUNT
    ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{self.save_counter}.pth")
    torch.save({
      "model_state_dict":model_state_dict,
      "optim_state_dict":optim_state_dict,
      "scheduler_state_dict":scheduler_state_dict,
      "save_counter":save_counter,
      "GLOBAL_STEP_COUNT":GLOBAL_STEP_COUNT,
      "episode_count":episode_count
    }, ckpt_path)
    self.update_index(ckpt_path, save_counter)
    print(f"SAVED CHECKPOINT TO {ckpt_path}")
    print(f"GLOBAL_STEP_COUNT: {GLOBAL_STEP_COUNT}\nEPISODE_COUNT: {episode_count}")

# let's try single-GPU training for now
class Learner:
  def __init__(self, weight_manager, batch_size, ckpt_dir, ckpt_freq, lr, restore=True, model_cfg=defaultGNN1Cfg):
    print("[Learner] INITIALIZING")
    self.frames = []
    self.buf = ReplayBuffer(logdir=ckpt_dir, batch_size=batch_size)
    self.weight_manager = weight_manager
    self.batch_size = batch_size
    self.ckpt_dir = ckpt_dir
    self.logdir = self.ckpt_dir
    self.ckpt_freq = ckpt_freq
    self.writer = SummaryWriter(log_dir=self.ckpt_dir)
    self.GLOBAL_STEP_COUNT = 0
    self.save_counter = 0
    self.model = GNN1(**model_cfg)
    if torch.cuda.is_available():
      print("[Learner] USING GPU")
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")
    self.model = self.model.to(self.device)
    self.lr = lr
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=[0.9, 0.98], eps=1e-9)

    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=1.0)
    self.batcher = Batcher()
    if restore:
      self.restore_weights()
    print("[Learner] LEARNER ONLINE")

  def restore_weights(self):
    ckpt = ray.get(self.weight_manager.load_latest_ckpt.remote())
    if ckpt is not None:
      self.set_weights(ckpt)

  def write_stats(self, stats):
    for name,value in stats.items():
      print(name, value)
      self.writer.add_scalar(name, value, self.GLOBAL_STEP_COUNT)

  def train_batch(self, batch):
    stats = train_step(self.model, self.optim, self.scheduler, self.batcher, *batch, device=self.device)
    self.write_stats(stats)

  def sync_weights(self, w):
    w.set_weights.remote(self.get_weights())
    print(f"SYNCED WEIGHTS TO WORKER {w}")

  def ingest_trajectory(self, tau):
    return self.buf.ingest_trajectory(tau)

  def set_episode_count(self, count):
    return self.buf.set_episode_count(count)

  def get_episode_count(self):
    return self.buf.get_episode_count()

  def get_weights(self):
    return self.model.state_dict()

  def set_weights(self, ckpt):
    self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    self.optim.load_state_dict(ckpt["optim_state_dict"])
    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    self.GLOBAL_STEP_COUNT = ckpt["GLOBAL_STEP_COUNT"]
    self.save_counter = ckpt["save_counter"]
    self.buf.set_episode_count(ckpt["episode_count"])

  def save_ckpt(self):
    episode_count = self.buf.get_episode_count()
    model_state_dict = deepcopy(self.model.state_dict())
    optim_state_dict = deepcopy(self.optim.state_dict())
    scheduler_state_dict = deepcopy(self.scheduler.state_dict())
    # if not self.weight_manager.gpu.remote():
    #   print("[Learner] PUTTING ON CPU")
    def to_cpu(d):
      for k,v in d.items():
        if isinstance(v, dict):
          d[k] = to_cpu(v)
        else:
          try:
            d[k] = v.cpu()
          except:
            pass
      return d
    model_state_dict = to_cpu(model_state_dict)
    optim_state_dict = to_cpu(optim_state_dict)
    scheduler_state_dict = to_cpu(scheduler_state_dict)
    ray.get(self.weight_manager.save_ckpt.remote(model_state_dict, optim_state_dict, scheduler_state_dict, self.save_counter+1, self.GLOBAL_STEP_COUNT, episode_count=episode_count))
    self.save_counter += 1

  def train2(self):
    OLD_GLOBAL_STEP_COUNT = self.GLOBAL_STEP_COUNT
    frames = self.frames
    while self.buf.frame_ready():
      frames.append(self.buf.get_frame())
    random.shuffle(frames)
    while len(frames) > self.batch_size:
      Gs = []
      mu_logitss = []
      actions = []
      gs = []
      advs = []
      masks = []
      cc_lengths = []
      episode_lengths = []
      for _ in range(self.batch_size):
        G, mu_logits, action, g, adv, mask, cc_length, episode_length = frames.pop()
        Gs.append(G)
        mu_logitss.append(mu_logits)
        actions.append(action)
        gs.append(g)
        advs.append(adv)
        masks.append(mask)
        cc_lengths.append(cc_length)
        episode_lengths.append(episode_length)
      batch = (Gs, mu_logitss, actions, gs, advs, masks, cc_lengths, episode_lengths)
      self.train_batch(batch)
      del batch
      self.GLOBAL_STEP_COUNT += 1
      if self.ckpt_freq is not None:
        if self.GLOBAL_STEP_COUNT % self.ckpt_freq == 0:
          self.save_ckpt()
    if self.GLOBAL_STEP_COUNT > OLD_GLOBAL_STEP_COUNT:
      self.scheduler.step()
    print("Finishing up and saving checkpoint.")
    self.save_ckpt()


  def train(self, step_limit=None, time_limit=None, synchronous=False):
    for param_group in self.optim.param_groups:
      param_group['lr'] = self.lr
    start = time.time()
    batches = []
    OLD_GLOBAL_STEP_COUNT = self.GLOBAL_STEP_COUNT
    try:
      sleep_count = 0
      while True:
        if step_limit is not None:
          if self.GLOBAL_STEP_COUNT > step_limit:
            print("STEP LIMIT", step_limit, "REACHED, STOPPING")
            break
        elif time_limit is not None:
          elapsed = time.time() - start
          if elapsed > time_limit:
            print("TIME LIMIT", time_limit, "REACHED, STOPPING")
            break
        else:
          pass

        if self.buf.batch_ready(self.batch_size):
          pass
        else:
          if synchronous:
            for _ in range(0):
              for batch in batches:
                self.train_batch(batch)
                self.GLOBAL_STEP_COUNT += 1
                if self.ckpt_freq is not None:
                  if self.GLOBAL_STEP_COUNT % self.ckpt_freq == 0:
                    self.save_ckpt()
            for b in batches:
              del b
            batches = []
            self.save_ckpt()
            break
          else:
            if sleep_count > 2:
              break
            SLEEP_INTERVAL = 0.5
            print(f"Replay buffer not ready. Sleeping for {SLEEP_INTERVAL}")
            time.sleep(SLEEP_INTERVAL)
            sleep_count += 1
            continue

        batch = self.buf.get_batch(self.batch_size)
        # batches.append(batch)

        self.train_batch(batch)
        del batch
        self.GLOBAL_STEP_COUNT += 1
        if self.ckpt_freq is not None:
          if self.GLOBAL_STEP_COUNT % self.ckpt_freq == 0:
            self.save_ckpt()

    finally:
      if self.GLOBAL_STEP_COUNT > OLD_GLOBAL_STEP_COUNT:
        self.scheduler.step()
      print("Finishing up and saving checkpoint.")
      self.save_ckpt()

def _parse_main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-workers",dest="n_workers",action="store", default=32, type=int)
  parser.add_argument("--n-epochs",dest="n_epochs",action="store", default=10, type=int)
  parser.add_argument("--cnfs", dest="cnfs",action="store")
  parser.add_argument("--time-limit", dest="time_limit", action="store", type=float)
  parser.add_argument("--lr", dest="lr", type=float, action="store", default=1e-4)
  parser.add_argument("--ckpt-dir", dest="ckpt_dir", action="store")
  parser.add_argument("--ckpt-freq", dest="ckpt_freq", action="store", type=int)
  parser.add_argument("--batch-size", dest="batch_size", action="store", type=int, default=32)
  parser.add_argument("--object-store", dest="object_store", action="store", default=None)
  parser.add_argument("--eps-per-worker", dest="eps_per_worker", action="store", default=25, type=int)
  parser.add_argument("--model-cfg", dest="model_cfg", action="store", default=None)
  parser.add_argument("--resample-frac", dest="resample_frac", action="store", default=0.4, type=float)
  parser.add_argument("--asynchronous", dest="asynchronous", action="store_true")

  opts = parser.parse_args()
  return opts

def _main():
  opts = _parse_main()
  if not os.path.exists(os.path.join(opts.ckpt_dir) + "/"):
    os.makedirs(os.path.join(opts.ckpt_dir) + "/")
  files = recursively_get_files(opts.cnfs, exts=["cnf","gz", "dimacs"], forbidden=["bz2"])
  print(f"TRAINING WITH {len(files)} CNFS")

  ray.init()

  WM_USE_GPU = False
  weight_manager = ray.remote(num_gpus=(1 if WM_USE_GPU else 0))(WeightManager).remote(ckpt_dir=opts.ckpt_dir)
  ray.get(weight_manager.load_latest_ckpt.remote())

  if opts.model_cfg is not None:
    with open(opts.model_cfg, "r") as f:
      model_cfg = json.load(f)
  else:
    print("[rl_lbd._main] warning: using default configuration")
    model_cfg = defaultGNN1Cfg

  learner = ray.remote(num_gpus=(1 if torch.cuda.is_available() else 0))(Learner).options(max_concurrency=(opts.n_workers+2)).remote(weight_manager=weight_manager, batch_size=opts.batch_size, ckpt_freq=opts.ckpt_freq, ckpt_dir=opts.ckpt_dir, lr=opts.lr, restore=True, model_cfg=model_cfg) # TODO: to avoid oom, either dynamically batch or preprocess the formulas beforehand to ensure that they are under a certain size -- this will requre some changes throughout to avoid a fixed batch size

  print("LEARNER ONLINE")
  ray.get(learner.restore_weights.remote())

  workers = [ray.remote(EpisodeWorker).remote(learner=learner, weight_manager=weight_manager, model_cfg=model_cfg) for _ in range(opts.n_workers)]

  pool = ActorPool(workers)

  for w in workers:
    ray.get(w.try_update_weights.remote())

  with open(os.path.join(opts.ckpt_dir, "log.txt"), "a") as f:
    print(f"[{datetime.datetime.now()}] STARTING TRAINING RUN", file=f)
    print("ARGS:", file=f)
    for k,v in vars(opts).items():
      print(f"    {k}  :  {v}", file=f)
    print("\n\n", file=f)

  def shuffle_environments(ws, resample_frac=1.0):
    for w in ws:
      resample = np.random.choice([True,False], p=[resample_frac, 1-resample_frac])
      if resample:
        ray.get(w.set_env.remote(from_file=random.choice(files)))
    print("shuffled environments")

  shuffle_environments(workers)
  for k_epoch in range(opts.n_epochs):
    if opts.asynchronous:
      train_handle = learner.train.remote(synchronous=False)
    waiting = 0
    completed = 0
    shuffle_environments(workers, opts.resample_frac)
    for _ in pool.map_unordered((lambda a,v: a.sample_trajectory.remote()), range(opts.eps_per_worker*opts.n_workers)):
      pass
    if opts.asynchronous:
      ray.get(train_handle)
    else:
      ray.get(learner.train.remote(synchronous=True))
      
if __name__ == "__main__":
    _main()
