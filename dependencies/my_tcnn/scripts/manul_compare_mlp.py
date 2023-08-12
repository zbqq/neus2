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
seed = 1377
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True




tcnn_mlp = tcnn.Network(
            # n_input_dims=self.encoder.n_output_dims,
        n_input_dims=64,
        # n_input_dims=128,
        n_output_dims=64,
        # n_output_dims=128,
        network_config={
            "otype": "CutlassMLP",
            "activation": "ReLU",
            # "activation": "None",
            "output_activation": "None",
            "n_neurons": 64,
            # "n_neurons": 128,
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
            # nn.Linear(128,128,bias=False),
            actvn,
            nn.Linear(64,64,bias=False),
            # nn.Linear(128,128,bias=False),
            actvn,
            nn.Linear(64,64,bias=False),
            # nn.Linear(128,128,bias=False),
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
# input = torch.randn((128,128), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.randn((1,128), dtype=torch.float).cuda().requires_grad_(True) 
input = torch.randn((1,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.randn((128,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.randn((256,128), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.randn((512,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.ones((64,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.ones((32,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.ones((128,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.ones((1,64), dtype=torch.float).cuda().requires_grad_(True) 
# input = torch.randn((1,64), dtype=torch.float).cuda().requires_grad_(True) 

y_tcnn = tcnn_mlp(input)
y = mlp(input)







loss = torch.ones_like(input, device=input.device)
# loss = torch.rand_like(input, device=input.device) ## not support, since identity.h not fully implemented!

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

nablas = autograd.grad(
    y,
    input,
    torch.ones_like(y, device=input.device),
    create_graph=True,
    retain_graph=True)[0]
nablas.backward(loss)

nablas_tcnn = autograd.grad(
    y_tcnn,
    input,
    torch.ones_like(y, device=input.device),
    create_graph=True,
    retain_graph=True)[0]

nablas_tcnn.backward(loss)



def is_relu(mlp_i):
    return 'ReLU' in str(mlp_i.__class__)

@torch.no_grad()
def dy_dx_dparams_more_fast(mlp, x, jacobian):
    # 0,2,4 : linear
    # 1,3: leakyRelu

    ## pytorch::Linear y = Wx + B

    ## get relu input
    relu_input = [] 
    output = x.clone()
    for i in range(len(mlp)):
        if is_relu(mlp[i]):
            relu_input.append(output)
        else:
            relu_input.append(None)
        output = mlp[i](output)
        
    hidden_layer_num = 0
    for i in range(len(mlp)):
        if is_relu(mlp[i]):
            continue
        else:
            hidden_layer_num += 1

    vector_front = [None for i in range(hidden_layer_num+2)]
    # tmp_front_vector = jacobian.sum(-1,keepdim=True)
    # tmp_front_vector = jacobian
    # tmp_front_vector = torch.ones((x.shape[0],64,1))
    tmp_front_vector = jacobian.unsqueeze(-1).expand(x.shape[0],jacobian.shape[1],1)
    hidden_layer_id = 0
    vector_front[hidden_layer_id] = tmp_front_vector
    hidden_layer_id +=1
    for i in range(len(mlp)):
        if is_relu(mlp[i]): 
            # # continue
            # import pdb
            # pdb.set_trace()
            set_zero_pos = relu_input[i] < 0 # set one row in a matrix
            # tmp_matrix[set_zero_pos] = 0
            # print(set_zero_pos.shape)
            # import pdb
            # pdb.set_trace()
            # import pdb
            # pdb.set_trace()
            # tmp_front_vector[set_zero_pos,:] = 0
            tmp_front_vector[set_zero_pos.unsqueeze(-1)] = 0
            vector_front[hidden_layer_id] = tmp_front_vector
            hidden_layer_id += 1
        else:
            # tmp_matrix = torch.matmul(mlp[i].weight, tmp_matrix)
            # tmp_matrix = torch.matmul(tmp_matrix, mlp[i].weight.T)
            tmp_front_vector = torch.matmul(mlp[i].weight, tmp_front_vector)


    vector_back = [None for i in range(hidden_layer_num+2)]
    # tmp_back_vector = torch.ones((1,mlp[-1].weight.shape[0]))
    tmp_back_vector = torch.ones((jacobian.shape[0],1,mlp[-1].weight.shape[0]))
    hidden_layer_id = hidden_layer_num + 1
    vector_back[hidden_layer_id] = tmp_back_vector
    hidden_layer_id -= 1
    for i in range(len(mlp)-1,-1,-1):
        if is_relu(mlp[i]):
            set_zero_pos = relu_input[i] < 0 # set one row in a matrix
            # tmp_matrix = tmp_matrix.T
            # tmp_back_vector[:,set_zero_pos] = 0
            tmp_back_vector[set_zero_pos.unsqueeze(1)] = 0
            # tmp_back_vector[set_zero_pos] = 0
            # tmp_matrix = tmp_matrix.T
            vector_back[hidden_layer_id] = tmp_back_vector
            hidden_layer_id -= 1
        else:
            # tmp_matrix = torch.matmul(tmp_matrix, mlp[i].weight)   
            # tmp_matrix = torch.matmul(tmp_matrix, mlp[i].weight)
            tmp_back_vector = torch.matmul(tmp_back_vector, mlp[i].weight)

    weight_gradient = []
    # import pdb
    # pdb.set_trace()
    hidden_layer_id = 1
    for i in range(len(mlp)):
        if is_relu(mlp[i]):
            weight_gradient.append(None)
            continue
        else:
            tmp_back_vector = vector_back[hidden_layer_id+1]
            tmp_front_vector = vector_front[hidden_layer_id-1]
            hidden_layer_id += 1
            # update_weight = torch.matmul(tmp_back_vector.T, tmp_front_vector.T)
            # update_weight = torch.matmul(tmp_back_vector.permute(0,2,1), tmp_front_vector.permute(0,2,1)).mean(0)
            update_weight = torch.matmul(tmp_back_vector.permute(0,2,1), tmp_front_vector.permute(0,2,1)).sum(0)
            weight_gradient.append(update_weight)

    return weight_gradient


dy_dx_dparams_result = dy_dx_dparams_more_fast(mlp,input.clone(),loss)


for i in range(len(mlp)):
    # if i == 0 or i == 2 or i ==4:
        # import pdb
        # pdb.set_trace()
    if 'ReLU' in str(mlp[i].__class__):
        continue
    # print("gt:", mlp[i].weight.grad.shape)
    # print("gt:", mlp[i].weight.grad)
    # print("pred:", dy_dx_dparams_result[i])
    # import pdb
    # pdb.set_trace()
    # exit(1)
    print("pytorch_equation error: ", (torch.abs(mlp[i].weight.grad - dy_dx_dparams_result[i])).mean())
    print("pytorch_equation relative error: ", torch.abs((mlp[i].weight.grad - dy_dx_dparams_result[i])/(mlp[i].weight.grad+1e-7)).mean())

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
