import torch
import random
import numpy as np
def setup():
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print("       Device", device)
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  torch.set_grad_enabled(True)
  torch.set_float32_matmul_precision("medium")
  seed = 1000
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  return device
