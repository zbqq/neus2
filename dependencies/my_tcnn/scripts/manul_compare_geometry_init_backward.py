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

def check(x,y):
    print("check error:",torch.abs(x-y).mean())
    print("check relative error:",(torch.abs(x-y)/(torch.abs(x)+1e-6)).mean())

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

config = {}
config["encoding"] = {
        # "otype": "HashGrid" if hash else "DenseGrid",
        "otype": "composite",
        "nested": [
        {
            "n_dims_to_encode": 3,
            "otype": "Grid",
            # "n_levels": 12,
            "n_levels": 16,
            "n_features_per_level": 2,
            # "log2_hashmap_size": 15,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.5,
            "interpolation": "Linear",
            # "loss_scale": 1.0,
        },
        {
            "otype": "Identity"
        }
        ]
    } 

config["network"] = {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            # "activation": "None",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
            # "n_hidden_layers": 0,
        }

model = tcnn.NetworkWithInputEncoding(
	6, 64,
	config["encoding"], config["network"]
)


encoder = tcnn.Encoding(3, {
        "otype": "HashGrid" if hash else "DenseGrid",
        # "n_levels": 12,
        "n_levels": 16,
        "n_features_per_level": 2,
        # "log2_hashmap_size": 15,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.5,
        "interpolation": "Linear",
        # "loss_scale": 1.0,
    })

actvn = nn.ReLU()


mlp = nn.Sequential(
            nn.Linear(48,64,bias=False),
            # nn.Linear(32,64,bias=False),
            actvn,
            nn.Linear(64,64,bias=False),
            actvn,
            nn.Linear(64,16,bias=False),
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
# param_sum = 0


nn.init.normal_(encoder.params,0.0,0.1)

encoder_param = encoder.params.numel()
model.params.data[-encoder_param:] = encoder.params.clone()
# param_sum = encoder.params.numel()
param_sum = 0
for i in range(len(mlp)):
    if 'ReLU' in str(mlp[i].__class__):
        continue
    param = mlp[i].weight
    nn.init.normal_(param, mean=0, std=0.01)
    # nn.init.constant_(param,0.03)
    model.params.data[param_sum:param_sum+param.numel()] = param.clone().detach().reshape(-1)
    param_sum+=param.numel()
    # nn.init.constant_(param,0.03)
    # nn.init.constant_(param,0.01)

# for name, param in tcnn_mlp.named_parameters():
    # nn.init.normal_(param, mean=0, std=0.01)
    # nn.init.constant_(param,0.03)
    # nn.init.constant_(param,0.01)




input = torch.randn((1,3), dtype=torch.half).cuda().requires_grad_(True)

x = encoder(input)
x.requires_grad_(True)
mlp_input = torch.cat([input,x.float(),torch.zeros([x.shape[0],13])],dim=-1)
y = mlp(mlp_input.float())[:,:1]

dy_dinput = autograd.grad(y,input,torch.ones_like(y, device=input.device),create_graph=True,retain_graph=True)[0]
dy_dmlp_input = autograd.grad(y,mlp_input,torch.ones_like(y, device=input.device),create_graph=True,retain_graph=True)[0]
dy_dx = autograd.grad(y,x,torch.ones_like(y, device=input.device),create_graph=True,retain_graph=True)[0]
dy_dx_direct_input = dy_dmlp_input[:,:3]
dy_dencoded_x = dy_dmlp_input[:,3:35]
# d_encodedx_dinput = autograd.grad(dy_dencoded_x,input,torch.ones_like(dy_dencoded_x, device=input.device),create_graph=True,retain_graph=True)[0]
d_encodedx_dinput = autograd.grad(x,input,dy_dencoded_x,create_graph=True,retain_graph=True)[0]
check(d_encodedx_dinput + dy_dx_direct_input, dy_dinput)
# y = mlp(x.float())[:,:1]

input = torch.randn((1,6), dtype=torch.half).cuda().requires_grad_(True)
y_tcnn = model(input)[:,:1]
# y_tcnn = model(torch.cat([input,input],dim=-1))[:,:1]
import pdb
pdb.set_trace()
# y = mlp(input)
# y = mlp(x.float())[:1,:]



#  autograd.grad(y,x,torch.ones_like(y, device=input.device),create_graph=True,retain_graph=True)[0]




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
# y.backward(torch.ones_like(y, device=input.device))

# y_tcnn.backward(loss)
# y_tcnn.backward(torch.ones_like(y_tcnn, device=input.device))
# import pdb
# pdb.set_trace()
nablas = autograd.grad(
    y,
    input,
    torch.ones_like(y, device=input.device),
    create_graph=True,
    retain_graph=True)[0]

loss = ((torch.linalg.norm(nablas.reshape(-1, 3), ord=2, dim=-1) - 1.0) ** 2).mean()
import time
start = time.time()
# nablas.backward(loss,retain_graph=True)
loss.backward(retain_graph=True)
end = time.time()
print('pytorch time:',(end-start))

autograd.grad(loss,nablas,torch.ones_like(loss, device=input.device),create_graph=True,retain_graph=True)[0]


normals = nablas
graident_norm = normals[:,0]*normals[:,0] + normals[:,1]*normals[:,1] + normals[:,2] * normals[:,2]
pos_gradient_norm_inv = 1 - 1 / ((graident_norm + 1e-10)**0.5)

dekloss_dgradient_ana = 2 * pos_gradient_norm_inv.unsqueeze(-1) * normals

nablas_tcnn = autograd.grad(
    y_tcnn,
    input,
    torch.ones_like(y, device=input.device),
    create_graph=True,
    retain_graph=True)[0]


loss_tcnn = ((torch.linalg.norm(nablas_tcnn.reshape(-1, 3), ord=2, dim=-1) - 1.0) ** 2)
dloss_dnablas = autograd.grad(loss_tcnn,nablas_tcnn,torch.ones_like(loss_tcnn, device=input.device),create_graph=True,retain_graph=True)[0]
loss_tcnn = loss_tcnn.mean()
# nablas_tcnn.backward(loss_tcnn)
start = time.time()
loss_tcnn.backward(retain_graph=True)
end = time.time()
print('tcnn time:',(end-start))

# exit(1)

def y2_from_x(x):
    return encoder(x)

# loss = torch.rand_like(x)

jacobian_y2_x = autograd.functional.jacobian(
    y2_from_x,
    input,
    # torch.tensor([1.0,2.0,3.0]),
    create_graph=True)



# mlp_i = 0
tcnn_i = 0
for mlp_i in range(len(mlp)):
    if not 'ReLU' in str(mlp[mlp_i].__class__):
        error = torch.abs((mlp[mlp_i].weight.grad - model.params.grad[:-encoder_param].reshape(layer_num,-1)[tcnn_i].reshape(mlp[mlp_i].weight.grad.shape))).mean()
        relative_error = torch.abs(((mlp[mlp_i].weight.grad - model.params.grad[:-encoder_param].reshape(layer_num,-1)[tcnn_i].reshape(mlp[mlp_i].weight.grad.shape)))/(mlp[mlp_i].weight.grad+1e-7)).mean()
        print('mlp[i].weight.grad',mlp[mlp_i].weight.grad)
        print('tcnn_mlp[i].weight.grad',model.params.grad[:-encoder_param].reshape(layer_num,-1)[tcnn_i].reshape(mlp[mlp_i].weight.grad.shape))
        print('error',error)
        print('relative_error',relative_error)
        tcnn_i += 1
    # mlp_i += 1



print("\noutput_error",torch.abs((y-y_tcnn)).mean())
print("output_relative_error",torch.abs(((y-y_tcnn)/(y+1e-7))).mean())

encoder.params.grad
model.params.grad[-encoder_param:]
encoder.params.grad
print("encoder_grad_error:",(encoder.params.grad-model.params.grad[-encoder_param:]).mean())
import pdb
pdb.set_trace()

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
        output = mlp[i](output)
        if is_relu(mlp[i]):
            relu_input.append(output)
        else:
            relu_input.append(None)
        
    hidden_layer_num = 0
    for i in range(len(mlp)):
        if is_relu(mlp[i]):
            continue
        else:
            hidden_layer_num += 1

    vector_front = [None for i in range(hidden_layer_num+2)]
    tmp_front_vector = jacobian.sum(-1,keepdim=True).float()
    # tmp_front_vector = jacobian
    # tmp_front_vector = torch.ones((x.shape[0],64,1))
    # tmp_front_vector = jacobian.unsqueeze(-1).expand(x.shape[0],jacobian.shape[1],1)
    hidden_layer_id = 0
    vector_front[hidden_layer_id] = tmp_front_vector
    hidden_layer_id +=1
    for i in range(len(mlp)):
        if is_relu(mlp[i]): 
            # # continue
            # import pdb
            # pdb.set_trace()
            set_zero_pos = relu_input[i] <= 0 # set one row in a matrix
            # set_zero_pos = relu_input[i] < 0 # set one row in a matrix
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
            tmp_front_vector = torch.matmul(mlp[i].weight, tmp_front_vector.float())


    vector_back = [None for i in range(hidden_layer_num+2)]
    # tmp_back_vector = torch.ones((1,mlp[-1].weight.shape[0]))
    tmp_back_vector = torch.ones((jacobian.shape[0],1,mlp[-1].weight.shape[0]))
    hidden_layer_id = hidden_layer_num + 1
    vector_back[hidden_layer_id] = tmp_back_vector
    hidden_layer_id -= 1
    for i in range(len(mlp)-1,-1,-1):
        if is_relu(mlp[i]):
            set_zero_pos = relu_input[i] <= 0 # set one row in a matrix
            # set_zero_pos = relu_input[i] < 0 # set one row in a matrix
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

new_vector = []
for i in range(len(loss)):
    new_vector.append(jacobian_y2_x[i,:,i,:] * loss[i])



# model.params[-encoder_param:]

# new_vector = torch.stack(new_vector,dim=0)
# dy_dx_dparams_result = dy_dx_dparams_more_fast(mlp,x.clone().float(),new_vector)

# for i in range(len(mlp)):
#     # if i == 0 or i == 2 or i ==4:
#         # import pdb
#         # pdb.set_trace()
#     if 'ReLU' in str(mlp[i].__class__):
#         continue
#     # print("gt:", mlp[i].weight.grad.shape)
#     # print("gt:", mlp[i].weight.grad)
#     # print("pred:", dy_dx_dparams_result[i])
#     # import pdb
#     # pdb.set_trace()
#     # exit(1)
#     print("pytorch_equation error: ", (torch.abs(mlp[i].weight.grad - dy_dx_dparams_result[i])).mean())
#     print("pytorch_equation relative error: ", torch.abs((mlp[i].weight.grad - dy_dx_dparams_result[i])/(mlp[i].weight.grad+1e-7)).mean())

