import sys
import pickle
import os
import argparse
import shutil
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
import torchio as tio

import tools

modalities = ('flair', 't1ce', 't1', 't2', 'seg')

# train
train_set = {
        'set': 'Train',
        'flist': 'train.txt',
        'has_label': True
        }

# test/validation data
valid_set = {
        'source': 'data/original',
        'target': 'data/small',
        'set': 'Valid',
        'flist': 'valid.txt',
        'has_label': True
        }

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data
    
def process(method, source, target, path, resample, zoom_val):
  source = os.path.join(source, path)
  target = os.path.join(target, path)
  target_dir = '/'.join(target.split('/')[:-1])
  if not os.path.exists( target_dir ):
    os.makedirs(target_dir)

  for modal in modalities:
    if method == 'zoom':
      data = nib.load(source + modal + '.nii').get_fdata()
      array = np.array(data).astype("uint8")
      new_array = zoom(array, zoom_val)
      img = nib.Nifti1Image(new_array, np.eye(4))
      img.to_filename(target + modal + '.nii')
      print(source + modal + '.nii')
    else:
      image = tio.Image(source + modal + '.nii')
      result = resample(image)
      result.save(target + modal + '.nii')
      print(source + modal + '.nii')

def doit(args, dset):
    resample = None
    if args.method=='resample':
      target_spacing = args.downsampling_factor / args.original_spacing  # in mm
      resample = tio.Resample(target_spacing)

    set = dset['set']
    file_list = os.path.join(args.source, dset['flist'])
    shutil.copy(file_list , os.path.join(args.target, dset['flist']))
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(set , sub, name + '_') for sub, name in zip(subjects, names)]
       
    for path in paths:
      process(method=args.method, source=args.source, target=args.target, path=path, zoom_val=args.zoom, resample=resample)

# START: PARAMETERS
# SAMPLE
# python utils/resample.py --source data/source_small/ --target data/small_test --method zoom --zoom 0.5
# python utils/resample.py --source data/source_small/ --target data/small_test --method resample --original_spacing 1 --downsampling_factor 2
parser = argparse.ArgumentParser()
parser.add_argument('--source', help='source directory', type=str)
parser.add_argument('--target', help='target directory', type=str)
parser.add_argument('--method', help='zoom or resample', type=str)
parser.add_argument('--zoom', help='zoom value sample: 0.5 or 2', type=float, required=False)
parser.add_argument('--original_spacing', help='original_spacing: 1', type=float, required=False)
parser.add_argument('--downsampling_factor', help='downsampling_factor: 2', type=float, required=False)

args = parser.parse_args()
# END PARAMETERS

if __name__ == '__main__':
  tools.print_now('START - ')
  if not os.path.exists( args.target ):
    os.makedirs(args.target)

  doit(args=args, dset=train_set)
  doit(args=args, dset=valid_set)
  tools.print_now('END - ')