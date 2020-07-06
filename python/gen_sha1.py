import os
import subprocess
from config import *
import uuid
import random

def sample_SHA1_problem(n_rounds, n_vars, msg, out_path):
  cgen_command = [CGEN_PATH, "encode", "sha1"]
  cgen_command += ["-r", str(n_rounds)]
  cgen_command += ["-v", f"M=string:{msg}"]
  cgen_command += [f"except:1..{n_vars}"]
  cgen_command += ["pad:sha1"]
  cgen_command += ["-vH=compute"]
  cgen_command += ["-mu"]
  cgen_command += [f"{out_path}"]

  result = subprocess.run(cgen_command, stdout=subprocess.PIPE).stdout.decode("UTF-8")

  print(result)

def gen_data(n_datapoints, outdir):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  for k in range(n_datapoints):
    name = str(uuid.uuid4())
    msg = name[:6]
    n_rounds = 17
    n_vars = random.choice(range(70,90))
    out_path = os.path.join(outdir, name+".cnf")
    sample_SHA1_problem(n_rounds, n_vars, msg, out_path)
    
if __name__ == "__main__":
  # # generate 200 training CNFs
  gen_data(200, os.path.join(PROJECT_DIR, "data", "sha1", "train"))
  # generate 50 test CNFs
  gen_data(50, os.path.join(PROJECT_DIR, "data", "sha1", "test"))
