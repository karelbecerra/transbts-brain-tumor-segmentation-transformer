import random
import numpy as np
from scipy import ndimage
import torch

class MaxMinNormalization(object):
  def __call__(self, sample):
    image = sample['image']
    label = sample['label']
    image_dim = sample['image_dim']
    Max = np.max(image)
    Min = np.min(image)
    image = (image - Min) / (Max - Min)

    return {'image': image, 'label': label, 'image_dim': image_dim}

class Random_Flip(object):
  def __call__(self, sample):
    image = sample['image']
    label = sample['label']
    image_dim = sample['image_dim']
    if random.random() < 0.5:
      image = np.flip(image, 0)
      label = np.flip(label, 0)
    if random.random() < 0.5:
      image = np.flip(image, 1)
      label = np.flip(label, 1)
    if random.random() < 0.5:
      image = np.flip(image, 2)
      label = np.flip(label, 2)

    return {'image': image, 'label': label, 'image_dim': image_dim}

class Random_Crop(object):
  def __call__(self, sample):
    image = sample['image']
    label = sample['label']
    image_dim = sample['image_dim']
    crop = image_dim.crop
    H = random.randint(0, image_dim.W - crop)
    W = random.randint(0, image_dim.H - crop)
    D = random.randint(0, image_dim.D - crop)

    image = image[H: H + crop, W: W + crop, D: D + crop, ...]
    label = label[..., H: H + crop, W: W + crop, D: D + crop]

    return {'image': image, 'label': label, 'image_dim': image_dim}

class Random_intencity_shift(object):
  def __call__(self, sample, factor=0.1):
    image = sample['image']
    label = sample['label']
    image_dim = sample['image_dim']

    scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
    shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

    image = image*scale_factor+shift_factor

    return {'image': image, 'label': label, 'image_dim' : image_dim}

class Random_rotate(object):
  def __call__(self, sample):
    image = sample['image']
    label = sample['label']
    image_dim = sample['image_dim']

    angle = round(np.random.uniform(-10, 10), 2)
    image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
    label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

    return {'image': image, 'label': label, 'image_dim': image_dim}

class Pad(object):
  def __call__(self, sample):
    image = sample['image']
    label = sample['label']
    image_dim = sample['image_dim']

    image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
    label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
    return {'image': image, 'label': label, 'image_dim': image_dim}

class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""
  def __call__(self, sample):
    image = sample['image']
    image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
    label = sample['label']
    label = np.ascontiguousarray(label)
    image_dim = sample['image_dim']

    image = torch.from_numpy(image).float()
    label = torch.from_numpy(label).long()

    return {'image': image, 'label': label, 'image_dim': image_dim}
