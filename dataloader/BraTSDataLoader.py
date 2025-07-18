import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from dataloader.augmentation import Pad, Random_Crop, Random_Flip, Random_intencity_shift, ToTensor
from utils.tools import pkload

def transform(sample):
  trans = transforms.Compose([
    Pad(), Random_Crop(), Random_Flip(), Random_intencity_shift(), ToTensor()
  ])
  return trans(sample)

def transform_valid(sample):
  trans = transforms.Compose([
    Pad(), Random_Crop(), ToTensor()
  ])
  return trans(sample)

class BraTSDataLoader(Dataset):
  def __init__(self, list_file, root='', mode='train', setname='t1ce_flair', image=None, setsize=-1):
    self.image = image
    self.setname = setname
    self.lines, paths, names = [], [], []
    with open(list_file) as f:
      for line in f:
        line = line.strip()
        name = line.split('/')[-1]
        names.append(name)
        path = os.path.join(root, line, name + '_')
        paths.append(path)
        self.lines.append(line)
    self.mode = mode
    if setsize > 0 and mode == 'train':
      self.names = names[:setsize]
      self.paths = paths[:setsize]
    else:
      self.names = names
      self.paths = paths

  def __getitem__(self, item):
    path = self.paths[item]
    image, label = pkload(path + self.setname + '.pkl')
    sample = {'image': image, 'label': label, 'image_dim' : self.image}
    sample = transform(sample) if self.mode == 'train' else transform_valid(sample)
    return sample['image'], sample['label']

  def __len__(self):
    return len(self.names)

  def collate(self, batch):
    return [torch.cat(v) for v in zip(*batch)]
