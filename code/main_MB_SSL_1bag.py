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
from nt_xent import NTXentLoss

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--nf', default=512, type=int, metavar='NF', help='num features')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=5e-3, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--file_path',
                    help='where to search for dataset .h5')                   
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--wd', type=str, default='./')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--norm', type=bool, default=False)
parser.add_argument('--folds', type=str)
parser.add_argument('--patience', type=int, default=100)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
loader_kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

print('Loading data')
def my_collate(batch):
  data1 = torch.cat([item[0] for item in batch])
  data2 = torch.cat([item[1] for item in batch])
  target = torch.Tensor([item[2] for item in batch])
  idxs1 = torch.cat([item[3] for item in batch])
  idxs2 = torch.cat([item[4] for item in batch])
  names = [item[5] for item in batch]
  
  return [data1, data2, target, idxs1, idxs2, names]

train_loader = data_utils.DataLoader(myDataset_MB_SSL_1bag(file_path=args.file_path,
                                                 dset="train",
                                                 norm=args.norm,
                                                 folds=args.folds),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     collate_fn=my_collate,
                                     **loader_kwargs)
valid_loader = data_utils.DataLoader(myDataset_MB_SSL_1bag(file_path=args.file_path,
                                                 dset="valid",
                                                 norm=args.norm,
                                                 folds=args.folds),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     collate_fn=my_collate,
                                     **loader_kwargs)
test_loader = data_utils.DataLoader(myDataset_MB_SSL_1bag(file_path=args.file_path,
                                               dset="test",
                                               norm=args.norm,
                                               folds=args.folds),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    collate_fn=my_collate,
                                    **loader_kwargs)
print('Training: '+str(args.batch_size*len(train_loader)))
print('Validation: '+str(args.batch_size*len(valid_loader)))
print('Testing: '+str(args.batch_size*len(test_loader)))

if args.model="gated":
  model = Attention_gated_MB_SSL(nf=args.nf)
else: # default
  model = Attention_MB_SSL(nf=args.nf)
if args.cuda:
  model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.reg)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0.000005)

train_losser=NTXentLoss('cuda', args.batch_size, 1.0, True)
valid_losser=NTXentLoss('cuda', args.batch_size, 1.0, True)
test_losser=NTXentLoss('cuda', args.batch_size, 1.0, True)

patience_counter=0
min_loss=1000
for epoch in range(1+start_epoch, args.epochs + 1):
  model.train()
  train_loss = 0.
  train_error = 0.
  for batch_idx, (xis, xjs, idxs, jdxs, n) in enumerate(train_loader):
    xis, xjs = xis.cuda(), xjs.cuda()
    xis, xjs = Variable(xis), Variable(xjs)

    optimizer.zero_grad()
    zis, zjs, Mi, Mj = model.calculate_stuff(xis, xjs, idxs, jdxs)
    
    loss=train_losser(zis,zjs)
    train_loss += loss.item()

    loss.backward()
    optimizer.step()

  train_loss /= len(train_loader)
  print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))

  model.eval()
  valid_loss = 0.
  valid_error = 0.
  with torch.no_grad():
    for batch_idx, (xis, xjs, idxs, jdxs, n) in enumerate(valid_loader):
      xis, xjs = xis.cuda(), xjs.cuda()
      xis, xjs = Variable(xis), Variable(xjs)

      zis, zjs, Mi, Mj = model.calculate_stuff(xis, xjs, idxs, jdxs)
      
      loss=valid_losser(zis,zjs)
      valid_loss += loss.item()

    valid_loss /= len(valid_loader)
    print(' Valid Set: Loss: {:.4f}'.format(valid_loss))

  if min_loss>valid_loss:
    min_loss=valid_loss
    patience_counter=0
    torch.save(model,args.wd+'best.model')
  else:
    patience_counter=patience_counter+1

  if patience_counter>args.patience:
    model=torch.load(args.wd+'best.model')
    if args.cuda:
      model.cuda()

    test_loss = 0.
    test_error = 0.
    with torch.no_grad():
      for batch_idx, (xis, xjs, idxs, jdxs, n) in enumerate(test_loader):
        xis, xjs = xis.cuda(), xjs.cuda()
        xis, xjs = Variable(xis), Variable(xjs)
        zis, zjs, Mi, Mj = model.calculate_stuff(xis, xjs, idxs, jdxs)
        
        loss=test_losser(zis,zjs)
        test_loss += loss.item()
   
    test_loss /= len(test_loader)
    print(' Test Set, Loss: {:.4f}'.format(test_loss))