from types import SimpleNamespace
import numpy as np
import time

import torch
import torch.optim
from torch.utils.data import DataLoader
from models.UnetTransformer import UnetTransformer
from models.criterions import softmax_dice

from dataloader.BraTSDataLoader import BraTSDataLoader
from utils.statusbar import print_status_bar
from utils import trace, args, modelfile, tools, dirs, torch_setup
from utils.tools import average

data_dir, load, batch, epochs, setname, with_scaler, setsize = args.read()

# DataSet Information
train_dir, train_file, valid_dir, valid_file, checkpoint_dir = dirs.read(data_dir=data_dir)


# Batch - 8 a 16
# Channels de 8 a 16
# Kernel - 3x3 hasta 5x5
# Learning. Rate 10-3 o 10-4
# Crop 128x128 hasta 80x80
# original_zoom_08 = (1, 192, 192, 124)
# original_zoom_07 = (1, 168, 168, 108)
# image = { 'W': 168, 'H': 168, 'D': 108, 'crop': 96 }

# Training Information
weight_decay=1e-5
lr = 0.004 #0.0002
crop = 112  # size of the image sample 
dim = 512   # transformer dimention
image = SimpleNamespace( W = 240, H = 240, D = 160, crop = crop)

def build_model():
  
  # Encoder & Decoder Channels
  enchannels = SimpleNamespace( b0=2, b1=8, b2=16, b3=32, b4=64 )
  dechannels = SimpleNamespace( b0=dim, b1=64, b2=32, b3=16, b4=8 )

  # Encoder Parameters
  emb_x = (crop//8)*(crop//8)*(crop//8) # 4096 # 
  
  encoder = SimpleNamespace( conv_patch_representation=True, positional_encoding_type="learned",
                            position_embeddings_x = emb_x, position_embeddings_y = 512, flatten_dim = crop * 16,
                            channels=enchannels)
  # Transformer parameters
  transformer = SimpleNamespace( heads=8, dropout_rate=0.1, attn_dropout_rate=0.1, hidden_dim = 4096, 
                                img_dim =crop, patch_dim = 8, num_layers=4, dim=dim )

  # Decoder Parameters
  decoder = SimpleNamespace( channels=dechannels )
  
  # Full Model Parameters
  parameters = SimpleNamespace( transformer=transformer, encoder=encoder, decoder=decoder, 
                               embedding_dim=dim, dropout_rate=0.1)

  model = UnetTransformer( parameters=parameters )
  return [1, 2, 3, 4], model

def validate(device, model, metrics, valid_loader):
  loss_list, dice1_list, dice2_list, dice3_list = [], [], [], []
  with torch.no_grad():
    model.eval()
    start_time = time.time()
    for j, data in enumerate(valid_loader):
      x, target = data
      x = x.to(device)
      target = target.to(device)
      output = model(x)
      loss, dice1, dice2, dice3 = softmax_dice(output, target)
      loss_list.append(loss.item())
      dice1_list.append(dice1.item())
      dice2_list.append(dice2.item())
      dice3_list.append(dice3.item())
      tmp_average = [average(loss_list), average(dice1_list), average(dice2_list), average(dice3_list)]
      print_status_bar(j+1, len(valid_loader), tmp_average, elapsed_time=time.time() - start_time, epoch=None, epochs=epochs)

    metrics['loss'].append( average(loss_list) )
    metrics['dice1'].append( average(dice1_list))
    metrics['dice2'].append( average(dice2_list))
    metrics['dice3'].append( average(dice3_list))
    return metrics

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
  for param_group in optimizer.param_groups:
    param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

def trainer():
  tools.print_separator()
  device = torch_setup.setup()

  # Load Data
  train_set = BraTSDataLoader(train_file, train_dir, mode='train', setname=setname, image=image, setsize=setsize)
  train_loader = DataLoader(dataset=train_set, batch_size=batch, drop_last=True, num_workers=2, pin_memory=True)

  valid_set = BraTSDataLoader(valid_file, valid_dir, mode='valid', setname=setname, image=image, setsize=setsize)
  valid_loader = DataLoader(dataset=valid_set, batch_size=1, drop_last=True, num_workers=2, pin_memory=True)

  # Load Model
  _, model = build_model()
  model.to(device)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
  scaler = torch.cuda.amp.GradScaler()

  start_epoch, min_loss, metrics_t, metrics_v = modelfile.load_model(dir=checkpoint_dir, load=load, model=model, optimizer=optimizer, scaler=scaler)  
  nbaches_ts = len(train_set)//batch

  tools.print_separator()
  for epoch in range(start_epoch, epochs):

    loss_list, dice1_list, dice2_list, dice3_list = [], [], [], []

    start_time = time.time()
    for i, data in enumerate(train_loader):
      adjust_learning_rate(optimizer, epoch, epochs, lr)
      x, target = data
      x = x.to(device)
      target = target.to(device)
      # Casts operations to mixed precision
      if with_scaler:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
          output = model(x)
          loss, dice1, dice2, dice3 = softmax_dice(output, target)
      else:
        output = model(x)
        loss, dice1, dice2, dice3 = softmax_dice(output, target)
      loss_list.append(loss.item())
      dice1_list.append(dice1.item())
      dice2_list.append(dice2.item())
      dice3_list.append(dice3.item())

      if with_scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        loss.backward()
        optimizer.step()
      optimizer.zero_grad(set_to_none=True)

      print_status_bar( i + 1, nbaches_ts, loss=[loss.item(), dice1.item(), dice2.item(), dice3.item()], 
                       elapsed_time = time.time() - start_time, epoch=epoch, epochs=epochs)

    metrics_t['loss'].append( average(loss_list) )
    metrics_t['dice1'].append( average(dice1_list) )
    metrics_t['dice2'].append( average(dice2_list) )
    metrics_t['dice3'].append( average(dice3_list) )

    #####conjunto de validacion #############
    metrics_v = validate(device=device, model=model, metrics=metrics_v, valid_loader=valid_loader)

    ##### trace metrics #########
    trace.trace( epoch=epoch, metrics_t=metrics_t, metrics_v=metrics_v )

    is_best = metrics_v['loss'][-1] < min_loss
    if is_best:      
      min_loss = metrics_v['loss'][-1]
    checkpoint = {  'epoch': epoch, 'min_loss': min_loss, 'metrics_t': metrics_t, 'metrics_v': metrics_v,  
                    'state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict(),
                    'scaler_dict': scaler.state_dict(), 
                  }
    modelfile.save_model(checkpoint=checkpoint, dir=checkpoint_dir, is_best=is_best)
      
  history = {}
  history['train'] = metrics_t
  history['valid'] = metrics_v
  return history

tools.print_now("START TRAINING -- ")
h = trainer()
tools.print_now("END TRAINING -- ")
trace.close()

his_file_name = 'history.npz'
np.savez(his_file_name, h = h, allow_pickle=True)
