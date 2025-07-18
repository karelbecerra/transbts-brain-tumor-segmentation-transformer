import sys
import pickle
import os
import numpy as np
import nibabel as nib

import tools

#modalities = ('flair', 't1ce', 't1', 't2')
modalities = ('flair', 't1ce')

# train
train_set = {
        'set': 'Train',
        'flist': 'train.txt',
        'has_label': True
        }

# test/validation data
valid_set = {
        'set': 'Valid',
        'flist': 'valid.txt',
        'has_label': True
        }

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data
    
def process(path, has_label=True, dtype='float32', setname='t1ceflair'):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(path + 'seg.nii'), dtype='uint8', order='C')
    images = np.stack([np.array(nib_load(path + modal + '.nii'), dtype=dtype, order='C') for modal in modalities], -1)

    output = path + setname + '.pkl'
    mask = images.sum(-1) > 0
    
    for k in range(len(modalities)):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return


def doit(dset, dtype, setname):
    set, has_label = dset['set'], dset['has_label']
    file_list = os.path.join(source, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(source, set , sub, name + '_') for sub, name in zip(subjects, names)]

    for path in paths:

        process(path, has_label, dtype=dtype, setname=setname)


# START: PARAMETERS
# SAMPLE
# python utils/preprocess.py "data/small_test" float32
source = sys.argv[1]
dtype = sys.argv[2]
setname = sys.argv[3]
# END PARAMETERS

if __name__ == '__main__':
    tools.print_now('START - ')
    doit(train_set, dtype=dtype, setname=setname)
    doit(valid_set, dtype=dtype, setname=setname)
    tools.print_now('END - ')