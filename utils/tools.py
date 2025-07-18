import torch.distributed as dist
import datetime
import pickle

start = None
parcial = None
def print_separator():
  print('##############################################################################')

def pkload(fname):
  with open(fname, 'rb') as f:
   return pickle.load(f)

def print_now(msg):
  global start
  global parcial
  now = datetime.datetime.now()
  if start is None:
     start = now
  if parcial is None:
     parcial = now
  elapsed = str(now - start).split(".")[0]
  parcial = str(now - parcial).split(".")[0]
  print( (msg + ' elapsed = {}  parcial = {}   now {} ').format(elapsed, parcial, now.strftime("%Y-%m-%d %H:%M:%S")) )
  parcial = datetime.datetime.now()

def average(list):
  return sum(list)/len(list)

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor