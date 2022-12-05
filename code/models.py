import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_MB_SSL(nn.Module):
     def __init__(self,nf):
          super(Attention_MB_SSL, self).__init__()
          self.L = int(nf)
          self.D = int(self.L/4)
          self.F = int(self.D/8)
          self.K = 1

          self.feature_extractor = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.L, self.D)
          )

          self.attention = nn.Sequential(
               nn.Linear(self.D, self.F),
               nn.Tanh(),
               nn.Linear(self.F, self.K)
          )

          self.projector = nn.Sequential(
               nn.Linear(self.D*self.K, self.F)
          )

          #self.loss = NTXentLoss('cuda', 45, 1.0, True)

     def forward(self, x, idxs):
          
          u=torch.unique(idxs)
          x = x.squeeze(0)
          H = self.feature_extractor(x)
          A = self.attention(H)
          A = torch.transpose(A, 1, 0)
          
          # By bag
          M = torch.empty((len(u),H.shape[1]), device='cuda')
          for a, i in enumerate(u):
               these=(idxs==i).nonzero(as_tuple=True)[0]
               A[:,these]=F.softmax(A[:,these], dim=1)
               M[a,:]=torch.mm(A[:,these],H[these,:])

          # By bag
          proj=torch.empty((u.shape[0],self.F), device='cuda')
          for a, i in enumerate(u):
               proj[a,:] = self.projector(M[a,:].unsqueeze(0))

          proj=F.normalize(proj,p=2,dim=1)

          return M, proj

     def calculate_stuff(self, xis, xjs, idxs, jdxs):
          Mi, zis = self.forward(xis, idxs)
          Mj, zjs = self.forward(xjs, jdxs)
          zis = torch.clamp(zis, min=1e-5, max=1. - 1e-5)
          zjs = torch.clamp(zjs, min=1e-5, max=1. - 1e-5)

          return zis, zjs, Mi, Mj

class Attention_gated_MB_SSL(nn.Module):
     def __init__(self,nf):
          super(Attention_gated_MB_SSL, self).__init__()
          self.L = int(nf)
          self.D = int(self.L/8)
          self.F = int(self.D/4)
          self.K = 1

          self.feature_extractor = nn.Sequential(nn.Linear(self.L, self.D), nn.ReLU(), nn.Dropout())
          self.attention_t = nn.Sequential(nn.Linear(self.D, self.F), nn.Tanh())
          self.attention_s = nn.Sequential(nn.Linear(self.D, self.F), nn.Sigmoid())
          self.attention = nn.Sequential(nn.Linear(self.F, self.K))
          self.projector = nn.Sequential(nn.Linear(self.D*self.K, self.F))

     def forward(self, x, idxs):
          
          u=torch.unique(idxs)
          H = self.feature_extractor(x)
          At = self.attention_t(H)
          As = self.attention_s(H)
          A = At.mul(As)
          A = self.attention(A)
          A = torch.transpose(A, 1, 0)
          
          # By bag
          M = torch.empty((len(u),H.shape[1]), device='cuda')
          for a, i in enumerate(u):
               these=(idxs==i).nonzero(as_tuple=True)[0]
               A[:,these]=F.softmax(A[:,these], dim=1)
               M[a,:]=torch.mm(A[:,these],H[these,:])

          # By bag
          proj=torch.empty((u.shape[0],self.F), device='cuda')
          for a, i in enumerate(u):
               proj[a,:] = self.projector(M[a,:].unsqueeze(0))

          proj=F.normalize(proj,p=2,dim=1)

          return M, proj

     def calculate_stuff(self, xis, xjs, idxs, jdxs):
          Mi, zis = self.forward(xis, idxs)
          Mj, zjs = self.forward(xjs, jdxs)
          zis = torch.clamp(zis, min=1e-5, max=1. - 1e-5)
          zjs = torch.clamp(zjs, min=1e-5, max=1. - 1e-5)

          return zis, zjs, Mi, Mj