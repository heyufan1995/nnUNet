import glob
import os
import copy
import numpy as np
import pdb

import shutil
import pickle
import json
from collections import OrderedDict
# copy to result if not converted yet
# shutil.copytree('/data/words/Task001_Word/', '/result/Task001_Word/')

words = '/result/nnUNet_raw_data/Task001_Word/'
with open(words + 'words.json', 'r') as f:
    data = json.load(f)
data_nnunet = copy.deepcopy(data)
data_nnunet['training'] = copy.deepcopy(data['training'] + data['validation'])

data_nnunet.pop('validation') 
data_nnunet['test'] = []
split = [OrderedDict()]

for idx, _ in enumerate(data_nnunet['training']):
    new_image = _['image'].replace('.nii.gz','_0000.nii.gz')
    try:
        shutil.move(words+_['image'], words+new_image)
    except:
        pass
    if 'Val' in _['image']:
        data_nnunet['training'][idx] = {'image': _['image'].replace('Val', 'Tr'), 'label':_['label'].replace('Val', 'Tr')}

for idx, _ in enumerate(data_nnunet['testing']):
    new_image = _['image'].replace('.nii.gz','_0000.nii.gz')
    try:
        shutil.move(words+_['image'], words+new_image)
    except:
        pass
    data_nnunet['test'].append(_['image'])

for idx, _ in enumerate(data['validation']):
    data['validation'][idx] = {'image': _['image'].replace('Val', 'Tr'), 'label':_['label'].replace('Val', 'Tr')}

# saved the split
split[0]['train'] = np.array([_['image'].split('/')[-1].replace('.nii.gz', '') for _ in data['training']])
split[0]['val'] = np.array([_['image'].split('/')[-1].replace('.nii.gz', '') for _ in data['validation']])

print(split)
with open('splits_final.pkl', 'wb') as f:
    pickle.dump(split, f)
# save the dataset json
with open('dataset.json', 'w') as f:
    json.dump(data_nnunet, f, indent=4)


