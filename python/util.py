import os
import re
import sys
import datetime

def timestamp():
  return str(datetime.datetime.now().strftime("%b-%d-%y--%H-%M-%S"))

def check_make_path(path):
  path = os.path.dirname(path)
  if not os.path.exists(path):
    try:
      os.makedirs(path)
    except:
      print("[WARNING] os.makedirs failed")

def files_with_extension(dir_path, ext=None):
  if ext is None:
    dataset_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path)]
  else:
    dataset_files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if re.search(r"." + ext, x)]
  return dataset_files

def recursively_get_files(folder, exts, forbidden=[]):
  files = []
  for r,d,f in os.walk(folder):
    for x in f:
      if any([re.search(r"." + ext, x) for ext in exts]):
        if any([re.search(r"." + ext, x) for ext in forbidden]):
          continue
        else:
          files.append(os.path.join(r,x))
  return files
