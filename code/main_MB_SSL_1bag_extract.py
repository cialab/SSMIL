from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from models import Attention_MB_SSL
from models import Attention_gated_MB_SSL
from dataloader import myDataset_MB_SSL_1bag

from scipy.io import savemat

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--file_path',
                    help='where to search for dataset .h5')                   
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='./models/1.model')
parser.add_argument('--wd', type=str, default='./')
parser.add_argument('--norm', type=bool, default=False)
parser.add_argument('--folds', type=str)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
loader_kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

print('Loading data')
def my_collate(batch):
  data1 = torch.cat([item[0] for item in batch])
  target = torch.Tensor([item[1] for item in batch])
  idxs1 = torch.cat([item[2] for item in batch])
  names = [item[3] for item in batch]
  
  return [data1, target, idxs1, names]
train_loader = data_utils.DataLoader(myDataset_MB_SSL_1bag_extract(file_path=args.file_path,
                                                 dset="train",
                                                 norm=args.norm,
                                                 folds=args.folds),
                                     batch_size=1,
                                     shuffle=False,
                                     collate_fn=my_collate,
                                     **loader_kwargs)
valid_loader = data_utils.DataLoader(myDataset_MB_SSL_1bag_extract(file_path=args.file_path,
                                                 dset="valid",
                                                 norm=args.norm,
                                                 folds=args.folds),
                                     batch_size=1,
                                     shuffle=False,
                                     collate_fn=my_collate,
                                     **loader_kwargs)
test_loader = data_utils.DataLoader(myDataset_MB_SSL_1bag_extract(file_path=args.file_path,
                                               dset="test",
                                               norm=args.norm,
                                               folds=args.folds),
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=my_collate,
                                    **loader_kwargs)
print('Training: '+str(len(train_loader)))
print('Validation: '+str(len(valid_loader)))
print('Testing: '+str(len(test_loader)))

model = torch.load(args.model)
if args.cuda:
  model.cuda()
model.eval()

with torch.no_grad():
  embeddings=[]
  names=[]
  labels=[]
  for batch_idx, (x, idxs, n) in enumerate(train_loader):
    x = x.cuda()
    x = Variable(x)
    M, z = model.forward(x, idxs)
    embeddings.append(M)
    names.append(n)

  embeddings=torch.cat(embeddings).cpu().numpy()
  names=[item for sublist in names for item in sublist]
  mdic = {'embeddings': embeddings, 'names': names}
  savemat(args.wd+'_train.mat',mdic)

  embeddings=[]
  names=[]
  labels=[]
  for batch_idx, (x, idxs, n) in enumerate(valid_loader):
    x = x.cuda()
    x = Variable(x)
    M, z = model.forward(x, idxs)
    embeddings.append(M)
    names.append(n)

  embeddings=torch.cat(embeddings).cpu().numpy()
  names=[item for sublist in names for item in sublist]
  mdic = {'embeddings': embeddings, 'names': names}
  savemat(args.wd+'_valid.mat',mdic)

  embeddings=[]
  names=[]
  labels=[]
  for batch_idx, (x, idxs, n) in enumerate(test_loader):
    x = x.cuda()
    x = Variable(x)
    M, z = model.forward(x, idxs)
    embeddings.append(M)
    names.append(n)

  embeddings=torch.cat(embeddings).cpu().numpy()
  names=[item for sublist in names for item in sublist]
  mdic = {'embeddings': embeddings, 'names': names}
  savemat(args.wd+'_test.mat',mdic)

