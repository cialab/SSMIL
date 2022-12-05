import pandas
from pathlib import Path
import torch
from torch.utils import data
from datetime import datetime
import random
import os
import numpy as np
import h5py

class myDataset_MB_SSL_1bag(data.Dataset):
    def __init__(self, file_path, dset):
        super(myDataset_MB_SSL_1bag,self).__init__()
        self.data_info = []
        self.dset=dset

        p = Path(file_path)
        if dset=="train":
            files = sorted(p.glob('training/*.h5'))
        if dset=="valid":
            files = sorted(p.glob('validation/*.h5'))
        if dset=="test":
            files = sorted(p.glob('testing/*.h5'))

        for f in files:
            name=str(f.resolve()).split('/')[-1].split('.')[0]
            self.data_info.append({'name': name, 'features': f.resolve()})

    def __getitem__(self, index):
        idx=index
        
        x = self.data_info[idx]['features']
        x = h5py.File(x,'r')[('features')]
        x = np.array(x).astype(np.float32)
        x = torch.from_numpy(x)
        ri = random.sample(range(0,x.shape[0]),round(x.shape[0]/4))
        rj = random.sample(range(0,x.shape[0]),round(x.shape[0]/4))
        xis = x[ri,:]
        xjs = x[ri,:]

        # idxs
        idxs = torch.Tensor([index]*xis.shape[0])
        jdxs = torch.Tensor([index]*xjs.shape[0])

        # get name
        name = self.data_info[idx]['name']
        
        return (xis, xjs, idxs, jdxs, name)

    def __len__(self):
        return len(self.data_info)

class myDataset_MB_SSL_1bag_extract(data.Dataset):
    def __init__(self, file_path, dset):
        super(myDataset_MB_SSL_1bag_extract,self).__init__()
        self.data_info = []
        self.dset=dset

        # pretty plz
        p = Path(file_path)
        if dset=="train":
            files = sorted(p.glob('training/*.h5'))
        if dset=="valid":
            files = sorted(p.glob('validation/*.h5'))
        if dset=="test":
            files = sorted(p.glob('testing/*.h5'))
        
        for f in files:
            name=str(f.resolve()).split('/')[-1].split('.')[0]
            self.data_info.append({'name': name, 'features': f.resolve()})

    def __getitem__(self, index):
        idx=index
        
        x = self.data_info[idx]['features']
        x = h5py.File(x,'r')[('features')]
        x = np.array(x).astype(np.float32)
        x = torch.from_numpy(x)

        # idxs
        idxs = torch.Tensor([index]*x.shape[0])

        # get name
        name = self.data_info[idx]['name']
        
        return (x,idxs,name)

    def __len__(self):
        return len(self.data_info)