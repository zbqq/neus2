import torch
import torch.nn as nn
from torch import autograd

# actvn = nn.ReLU()
# actvn = nn.LeakyReLU()
actvn = nn.Softplus(beta=100)

mlp = nn.Sequential(
            nn.Linear(3,64,bias=False),
            # nn.Linear(32,64,bias=False),
            actvn,
            nn.Linear(64,64,bias=False),
            actvn,
            nn.Linear(64,64,bias=False),
            # actvn,
            # nn.Linear(64,64,bias=False),
            # actvn,
            # nn.Linear(64,64,bias=False)
        ).cuda()

x = torch.rand((3,3)).cuda()
x.requires_grad_(True)

y = mlp(x)[:,:1]

dy_dx = autograd.grad(y,x,torch.ones_like(y, device=x.device),create_graph=True,retain_graph=True)[0]
print(y.shape)
print(dy_dx.shape)

ddydx_dx = autograd.grad(dy_dx,x,torch.ones_like(dy_dx, device=x.device),create_graph=True,retain_graph=True)[0]
print(ddydx_dx)