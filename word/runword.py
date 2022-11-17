import glob
import os
import copy
import numpy as np
import pdb
words = '/data/words/Task001_Words/'
import shutil
import json
from collections import OrderedDict
if os.path.exists('split_final.pkl'):
    with open('words.json', 'r') as f:
        data = json.load(f)
    data_nnunet = copy.deepcopy(data)
    data_nnunet['training'] = data['training'] + data['validation'] 
    split = [OrderedDict()]

    for idx, _ in enumerate(data_nnunet['training']):
        new = _.replace('.nii.gz','_0000.nii.gz')
        shutil.move(words+_, words+new)
        data_nnunet['training'][idx] = new

    # saved the split
    split[0]['train'] = np.array([_.split('/')[-1].replace('.nii.gz', '') for _ in data['training']])
    split[0]['val'] = np.array([_.split('/')[-1].replace('.nii.gz', '') for _ in data['training']])
    with open('split_final.pkl', 'w') as f:
        np.save(f, split)
    # save the dataset json
    with open('dataset.json', 'w') as f:
        json.dump(data_nnunet, f)
else:
    print('already preprocessed')


