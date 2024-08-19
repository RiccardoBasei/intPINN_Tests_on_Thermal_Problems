import numpy as np
import matplotlib.pyplot as plt
from create_parameters import create_parameters
from create_data import data_from_ode
from custom_resnet_parallel_training import CustomResNet, parallel_train
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.io import loadmat
from data_import import normalization
import pandas as pd
from interp1 import interp1

plt.close('all')

# import material lookup tables
k_Si_data = pd.read_csv('non_lin_data/LUT_variables/k_Si_fun1.csv')
k_cer_data = pd.read_csv('non_lin_data/LUT_variables/k_ceramic_fun1.csv')
cp_Si_data = pd.read_csv('non_lin_data/LUT_variables/cp_Si_fun1.csv')
mat_data = [k_Si_data, k_cer_data, cp_Si_data]

# interpolation of data np
k_Si  = lambda T_Si: np.interp(T_Si, k_Si_data['T'].values, k_Si_data['k_Si'].values)
k_cer = lambda T_cer: np.interp(T_cer, k_cer_data['T'].values, k_cer_data['k_cer'].values)
cp_Si = lambda T_Si: np.interp(T_Si, cp_Si_data['T'].values, cp_Si_data['cp_Si'].values)

param_fT = lambda T: [k_Si(T[3]), k_cer(T[2]), cp_Si(T[3])]

#inerpolation of data torch
k_Si_T  = lambda T_Si: interp1(torch.Tensor(k_Si_data['T'].values), torch.Tensor(k_Si_data['k_Si'].values), T_Si)
k_cer_T = lambda T_cer: interp1(torch.Tensor(k_cer_data['T'].values), torch.Tensor(k_cer_data['k_cer'].values), T_cer)
cp_Si_T = lambda T_Si: interp1(torch.Tensor(cp_Si_data['T'].values), torch.Tensor(cp_Si_data['cp_Si'].values), T_Si)

param_fT_tensor = lambda T: [k_Si_T(T[3]), k_cer_T(T[2]), cp_Si_T(T[3])]

# plot the material characteristics
plt.figure(1)
Temp = torch.linspace(200,800,150)
plt.plot(Temp.detach().numpy(),k_Si_T(Temp).detach().numpy())
plt.plot(k_Si_data['T'], k_Si_data['k_Si'],'ro')
plt.xlabel('Temperature [K]')
plt.ylabel('k_Si [W/(m K)]')
plt.grid(linewidth=.5)
plt.figure(2)
Temp = np.linspace(200,800,150)
plt.plot(Temp,k_cer(Temp))
plt.plot(k_cer_data['T'], k_cer_data['k_cer'],'ro')
plt.xlabel('Temperature [K]')
plt.ylabel('k_cer [J/(kg K)]')
plt.grid(linewidth=.5)
plt.figure(3)
Temp = np.linspace(200,1900,150)
plt.plot(Temp,cp_Si(Temp))
plt.plot(cp_Si_data['T'], cp_Si_data['cp_Si'],'ro')
plt.xlabel('Temperature [K]')
plt.ylabel('cp_Si [W/(m K)]')
plt.grid(linewidth=.5)

# Define the path to your .mat file
mat_file_path = 'non_lin_data/rom_PE_simple.mat'

# Load the .mat file
data = loadmat(mat_file_path, mat_dtype=True)
rom = {'A': data['rom']['A'][0][0][0],
       'B': data['rom']['B'][0][0],
       'C': data['rom']['C'][0][0],
       'E': data['rom']['E'][0][0][0]
       }

# data creation
parameters = create_parameters()

# normalization of the system
rom_norm = normalization(rom, parameters)

# display some info
num_outputs = np.shape(rom_norm['C'])[0]
num_inputs = np.shape(rom_norm['B'])[1]-1 # I remove ambient temp as an input
num_states = np.shape(rom_norm['A'][0])[0]
print("Info about the non-linear system")
print(f" Number of states = {num_states}")
print(f" Number of inputs = {num_inputs}")
print(f" Number of outputs = {num_outputs}")

# datasets for the training
print('Dataset creation...')
training_dataset, steady_state_dataset = data_from_ode(parameters, rom_norm, param_fT)
print('Dataset created successfully!')
        
training_datasets = {'ode_random_dataset': training_dataset,
                     'steady_state_dataset': steady_state_dataset}

# network definition
layers = [num_states+2, 200, 200, 200, 200, 200, num_states]
model = CustomResNet(layers)
torch.manual_seed(0)
print('Network initialized')

# dynamical system preparation for training
sys_info = {'param': param_fT_tensor, 'sys_matrices': rom_norm}

# target net definition
model1 = CustomResNet(layers)
model1.load_state_dict(model.state_dict())

# training options
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, cooldown=25, patience=25)
loss_fn = nn.MSELoss()
weights = [5,1,0]    

# training
[model, train_loss_log] = parallel_train(
    model,
    model1,
    training_datasets,
    parameters,
    loss_fn,
    weights,
    optimizer,
    scheduler,
    sys_info,
    shuffle_dataset=True)

# save the model
net_state_dict = model.state_dict()
print(net_state_dict.keys())
torch.save(net_state_dict, 'model1.torch')

# plot training curve and save loss vector
plt.figure()
plt.semilogy(range(len(train_loss_log)),train_loss_log)

# show all the plots
plt.show()