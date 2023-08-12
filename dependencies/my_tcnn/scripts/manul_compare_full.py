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
# # torch.set_default_tensor_type(torch.cuda.HalfTensor)
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
        "n_features_per_level": 4,
        # "log2_hashmap_size": 15,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.5,
        "interpolation": "Linear"
    })

tcnn_mlp = tcnn.Network(
            # n_input_dims=self.encoder.n_output_dims,
        n_input_dims=64,
        # n_input_dims=32,
        n_output_dims=64,
        network_config={
            "otype": "CutlassMLP",
            "activation": "ReLU",
            # "activation": "None",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
            # "n_hidden_layers": 0,
        },
    )

# tcnn_mlp.params

actvn = nn.ReLU()



# mlp_start = nn.Sequential(nn.Linear(3,9),
#             nn.Linear(9,9),
#             nn.Linear(9,9),
#             nn.Linear(9,9),
#             # nn.Linear(9,3))
#             nn.Linear(9,9))

mlp = nn.Sequential(
            nn.Linear(64,64,bias=False),
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

layer_num = 0
for i in range(len(mlp)):
    if 'ReLU' in str(mlp[i].__class__):
        continue
    layer_num += 1
# for i in range(2):
    # nn.init.normal_(mlp[i].weight, mean = 0, std=0.01)
    # tcnn_mlp.params.data[:1024] = mlp[i].weight.clone().detach().reshape(-1)
    # tcnn_mlp.params.reshape(2,-1)[i].reshape(mlp[i].weight.shape) = mlp[i].weight.copy()
    # nn.init.normal_(tcnn_mlp.params.reshape(2,-1)[i].reshape(mlp[i].weight.shape), mean = 0, std=0.01)
# param_sum = 0
# for name, param in mlp.named_parameters():
#     nn.init.normal_(param, mean=0, std=0.01)
#     tcnn_mlp.params.data[param_sum:param_sum+param.numel()] = param.clone().detach().reshape(-1)
#     param_sum+=param.numel()
param_sum = 0
for i in range(len(mlp)):
    if 'ReLU' in str(mlp[i].__class__):
        continue
    param = mlp[i].weight
    nn.init.normal_(param, mean=0, std=0.01)
    # nn.init.constant_(param,0.03)
    tcnn_mlp.params.data[param_sum:param_sum+param.numel()] = param.clone().detach().reshape(-1)
    param_sum+=param.numel()
    # nn.init.constant_(param,0.03)
    # nn.init.constant_(param,0.01)

# for name, param in tcnn_mlp.named_parameters():
    # nn.init.normal_(param, mean=0, std=0.01)
    # nn.init.constant_(param,0.03)
    # nn.init.constant_(param,0.01)



# input = torch.tensor([[0.17, 0.55, 0.79]], dtype=torch.float).cuda().requires_grad_(True)
# input = torch.randn((1,16), dtype=torch.double).cuda().requires_grad_(True)
# input = torch.zeros((1,16), dtype=torch.double).cuda().requires_grad_(True)
# input = torch.ones((1,64), dtype=torch.double).cuda().requires_grad_(True)
# input = torch.ones((1,64), dtype=torch.half).cuda().requires_grad_(True)
# input = torch.randn((1,64), dtype=torch.double).cuda().requires_grad_(True) 
# input = torch.randn((1,64), dtype=torch.half).cuda().requires_grad_(True) 
# input = torch.randn((128,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.randn((256,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.ones((128,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.ones((1,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.randn((1,64), dtype=torch.float).cuda().requires_grad_(True) 

# input = torch.randn((1,3), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.randn((1,3), dtype=torch.half).cuda().requires_grad_(True) 
input = (torch.randn((1024,3), dtype=torch.half) * 100).cuda().requires_grad_(True)
x = encoder(input)
x_tcnn = encoder(input.clone())

# y_tcnn = tcnn_mlp(input)
y_tcnn = tcnn_mlp(x_tcnn)
# y = mlp(input)
y = mlp(x.float())







loss = torch.ones_like(input, device=input.device)

# y.backward(loss)
# y_tcnn.backward(loss)

# # mlp[0].weight.grad.shape
# # mlp[1].weight.grad.shape

# ## check grad
# for i in range(2):
#     error = torch.abs((mlp[i].weight.grad - tcnn_mlp.params.grad.reshape(2,-1)[i].reshape(mlp[i].weight.grad.shape))).mean()
#     print(error)
# import pdb
# pdb.set_trace()


## check double grad

# y.backward(loss)
y.backward(torch.ones_like(y, device=input.device))

# y_tcnn.backward(loss)
y_tcnn.backward(torch.ones_like(y_tcnn, device=input.device))

# mlp_i = 0
tcnn_i = 0
for mlp_i in range(len(mlp)):
    if not 'ReLU' in str(mlp[mlp_i].__class__):
        error = torch.abs((mlp[mlp_i].weight.grad - tcnn_mlp.params.grad.reshape(layer_num,-1)[tcnn_i].reshape(mlp[mlp_i].weight.grad.shape))).mean()
        relative_error = torch.abs(((mlp[mlp_i].weight.grad - tcnn_mlp.params.grad.reshape(layer_num,-1)[tcnn_i].reshape(mlp[mlp_i].weight.grad.shape)))/(mlp[mlp_i].weight.grad+1e-7)).mean()
        print('mlp[i].weight.grad',mlp[mlp_i].weight.grad)
        print('tcnn_mlp[i].weight.grad',tcnn_mlp.params.grad.reshape(layer_num,-1)[tcnn_i].reshape(mlp[mlp_i].weight.grad.shape))
        print('error',error)
        print('relative_error',relative_error)
        tcnn_i += 1
    # mlp_i += 1



print("\noutput_error",torch.abs((y-y_tcnn)).mean())
print("output_relative_error",torch.abs(((y-y_tcnn)/(y+1e-7))).mean())
