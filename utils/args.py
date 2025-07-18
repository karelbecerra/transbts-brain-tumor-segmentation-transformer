import sys
import argparse

# START: PARAMETERS
def read():
  # PARAMETERS SAMPLE
  # python LocalTransBTS.py --data_dir=data/small_full --load no --batch 16 --epochs 100 --setname t1ce_flair --scaler yes
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', help='source directory', type=str)
  parser.add_argument('--load', help="load latest model 'yes' or 'no'", type=str)
  parser.add_argument('--batch', help='batch size', type=int)
  parser.add_argument('--epochs', help='number of epochs', type=int)
  parser.add_argument('--setname', default='t1ce_flair', help='float16 or float32', type=str)
  parser.add_argument('--scaler', default='no', help="include scaler 'yes' or 'no'", type=str)
  parser.add_argument('--setsize', default=-1, help="limit training datase", type=int)
  args = parser.parse_args()
  # END PARAMETERS
  data_dir = args.data_dir
  load = args.load in ("yes", "true", "t", "1", "True")
  batch = args.batch
  epochs = args.epochs
  setname = args.setname
  setsize = args.setsize
  with_scaler = args.scaler in ("yes", "true", "t", "1", "True")
  return data_dir, load, batch, epochs, setname, with_scaler, setsize
  # END PARAMETERS
