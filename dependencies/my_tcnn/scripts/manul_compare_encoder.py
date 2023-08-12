import pdb
import torch
import torch.nn as nn
from torch import autograd
from torch.optim import Adam
import torch.nn.functional as F

import sys

import random
import numpy as np
import os
import tinycudann as tcnn

# torch.set_default_tensor_type(torch.cuda.DoubleTensor)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.set_default_tensor_type(torch.cuda.HalfTensor)
# seed = 1377
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True


encoder = tcnn.Encoding(3, {
        "otype": "HashGrid" if hash else "DenseGrid",
        # "n_levels": 12,
        "n_levels": 16,
        "n_features_per_level": 2,
        # "log2_hashmap_size": 15,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.5,
        "interpolation": "Linear"
    })

# nn.init.constant_(encoder.params,1.0)
# nn.init.constant_(encoder.params,0.1)
seed = 1377
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# nn.init.normal_(encoder.params,0,0.1)
# nn.init.uniform_(encoder.params,0,1.0)
nn.init.constant_(encoder.params,1.0)

# x = (torch.tensor([[0.17, 0.55, 0.79]], dtype=torch.float)).requires_grad_(True).cuda()
# x = (torch.tensor([[1.0,1.0,1.0]], dtype=torch.float)).requires_grad_(True).cuda()
# x = (torch.ones((128,3), dtype=torch.float)).requires_grad_(True).cuda()
# x = (torch.ones((1,3), dtype=torch.float)).requires_grad_(True).cuda()
x = (torch.randn((128,3), dtype=torch.float)).requires_grad_(True).cuda()
# x = (torch.randn((1,3), dtype=torch.float)).requires_grad_(True).cuda()

# y = encoder(x)

def y2_from_x(x):
    return encoder(x)

loss = torch.ones_like(x)
# loss = torch.rand_like(x)

jacobian_y2_x = autograd.functional.jacobian(
    y2_from_x,
    x,
    # torch.tensor([1.0,2.0,3.0]),
    create_graph=True)

print("jacobian:",jacobian_y2_x)
# import pdb
# pdb.set_trace()
y = encoder(x)

print('y.shape',y.shape)
print('y',y)

# dy = (jacobian_y2_x[0,:,0,:] * loss[0]).sum(-1)
dy = []
for i in range(len(loss)):
    dy.append((jacobian_y2_x[i,:,i,:] * loss[i]).sum(-1))
dy = torch.stack(dy,dim=0)

vis_index = [0]
print('dy.shape',dy.shape)
# print('dy',dy[vis_index])


dy_my = encoder.cal_dy(loss)

print('dy_my.shape',dy_my.shape)
print('dy_my',dy_my[vis_index])
print('dy-y',(dy-dy_my)[vis_index])

print('error:',torch.abs((dy-dy_my))[vis_index].mean())
print('relative_error:',torch.abs((dy-dy_my)/dy)[vis_index].mean())
print('relative_error:',torch.abs((dy-dy_my)/(dy+1e-5).mean()))

import pdb
pdb.set_trace()