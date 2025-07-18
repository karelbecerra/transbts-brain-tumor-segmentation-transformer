import os
import time
import glob

import torch
import numpy as np

last_model = 'last_model.pth'

def get_best_model(path):
  files = glob.glob(path + '*.pth')
  return max(files, key=os.path.basename) if len(files) > 0 else 'xxxxxx'

def load_model(dir, load, model, optimizer, scaler ):
  if not os.path.exists(dir):
    os.makedirs(dir)

  file_name = os.path.join(dir, last_model)
  start_epoch = 0
  min_loss = np.inf
  metrics_t, metrics_v = {}, {}
  metrics_t['loss'], metrics_t['dice1'], metrics_t['dice2'], metrics_t['dice3'] = [], [], [], []
  metrics_v['loss'], metrics_v['dice1'], metrics_v['dice2'], metrics_v['dice3'] = [], [], [], []

  if load and os.path.isfile(file_name):  
    print('       Loading saved checkpoint {}'.format(file_name))
    checkpoint = torch.load(file_name, map_location=lambda storage, loc: storage)
    min_loss = checkpoint['min_loss']
    metrics_t = checkpoint['metrics_t']
    metrics_v = checkpoint['metrics_v']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer_dict"])
    scaler.load_state_dict(checkpoint["scaler_dict"])
    start_epoch = checkpoint['epoch'] + 1
    print('         min_loss {} in epoch {} '.format(min_loss, checkpoint['epoch']))
  else:
    print('       Start fresh training')
  return start_epoch, min_loss, metrics_t, metrics_v

def save_model(checkpoint, dir, is_best):
  file_name = os.path.join(dir, last_model)
  torch.save( checkpoint, file_name)
  if is_best:
    print('       #### NEW BEST MODEL ####')
    version = time.strftime("%Y_%m_%d_%Hh_%Mm_%Ss", time.localtime())
    name = 'best_model_epoch_' + version + '.pth'
    file_name = os.path.join(dir, name)
    torch.save( checkpoint, file_name)
