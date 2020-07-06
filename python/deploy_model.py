from gnn import *
from pathlib import Path

def load_GNN1_drat(model_cfg_path, ckpt_path):
  with open(model_cfg_path, "r") as f:
    cfg = json.load(f)
  model = GNN1_drat(**cfg)

  if ckpt_path is not None:
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
      print("GPU AVAILABLE")
    print("LOADING CKPT FROM", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    try:
      model.load_state_dict(ckpt["model_state_dict"], strict=False)
    except:
      model.load_state_dict(ckpt["models"][0], strict=False)
  else:
    print("WARNING: serializing randomly initialized network")

  model.eval()
  return model

def serialize_GNN1_drat(model, save_path):
  m = torch.jit.script(model)
  m.save(save_path)

def deploy_GNN1_drat(model_cfg_path, ckpt_path, save_path):
  serialize_GNN1_drat(load_GNN1_drat(model_cfg_path, ckpt_path), save_path)

def _parse_main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--cfg", dest="cfg", action="store")
  parser.add_argument("--ckpt", dest="ckpt_path", type=str, action="store")
  parser.add_argument("--dest", dest="dest", type=str, action="store")
  opts = parser.parse_args()
  return opts

def _main():
  opts = _parse_main()
  ckpt_path = opts.ckpt_path
  rootname = str(Path(opts.cfg).stem)
  save_path_drat = os.path.join(opts.dest, rootname + "_drat.pt")
  deploy_GNN1_drat(model_cfg_path=opts.cfg, ckpt_path=ckpt_path, save_path=save_path_drat)
  print(f"Saved model to {save_path_drat}")

if __name__ == "__main__":
  _main()
