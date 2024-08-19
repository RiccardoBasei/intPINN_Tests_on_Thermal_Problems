import numpy as np
import matplotlib.pyplot as plt
from create_parameters import create_parameters
from create_data import data_from_ode, power
from custom_resnet_parallel_training import CustomResNet, parallel_train
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.io import loadmat
import pandas as pd
import time

plt.close('all')

# Define the path to your .mat file
mat_file_path = 'thermal_data/python/python_data/matlab_data.mat'

# Load the .mat file
data = loadmat(mat_file_path, mat_dtype=True)
rom = data['rom']

# linear system
Ad = rom['A'][0][0]
Bd = rom['B'][0][0]
Cd = rom['C'][0][0]
Ed = rom['E'][0][0]

# load material characteristics
cp_Si_data = pd.read_csv('thermal_data/LUT_variables/cp_Si_fun1.csv')
cp_Si = lambda T_Si: np.interp(T_Si, cp_Si_data['T'].values, cp_Si_data['cp_Si'].values)
R = lambda T: 1 + 4e-3*(T-300)

# display some info
print("Info about the non-linear system")
print(" Number of states = " +str(np.shape(Ad)[0]))
print(" Number of inputs = " +str(np.shape(Bd)[1]-1))
print(" Number of outputs = " +str(np.shape(Cd)[0]))
num_outputs = np.shape(Cd)[0]
num_inputs = np.shape(Bd)[1]-1 # I remove ambient temp as an input
num_states = np.shape(Ad)[0]

# data creation
parameters = create_parameters()


# dynamical system normalized definition
xmax = np.array([parameters['intervals']['x0_1'][1],
                 parameters['intervals']['x0_2'][1],
                 parameters['intervals']['x0_3'][1]])
xmin = np.array([parameters['intervals']['x0_1'][0],
                 parameters['intervals']['x0_2'][0],
                 parameters['intervals']['x0_3'][0]])
normx = np.diag(xmax - xmin)
dtmax = parameters['intervals']['delta_t'][1]
umax = parameters['intervals']['p'][1]

dynamical_system_normalized = {
    'A': np.dot(torch.Tensor(Ad), normx),
    'B': Bd*umax,
    'E': np.dot(Ed, normx)/dtmax,
    'C': np.dot(Cd, normx),
    'D': np.dot(Cd, xmin),
    'K': np.dot(Ad, xmin),
    'f': lambda T: 1 + 7e-4*cp_Si(T),
    'coeff': lambda T: R(T)
    }

# dataset generation
print('Dataset creation...')
training_dataset, steady_state_dataset = data_from_ode(parameters, dynamical_system_normalized, power)
training_datasets = {'ode_random_dataset': training_dataset,
                     'steady_state_dataset': steady_state_dataset}
print('Dataset created.')

# network definition
layers = [num_states+2, 30, 30, 30, 30, 30, 30, num_states]
model = CustomResNet(layers)
torch.manual_seed(0)
print('Network initialized')

# target net definition
model1 = CustomResNet(layers)
model1.load_state_dict(model.state_dict())

################################## first training step ########################
parameters['training']['epoch_schedule'] = [1500]
parameters['training']['batching_schedule'] = [100]

# training
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, cooldown=25, patience=70)
loss_fn = nn.MSELoss()
weights = [700, 100, 800]

# training loop
start = time.time()
[model, train_loss_log] = parallel_train(
    model,
    model1,
    training_datasets,
    parameters,
    loss_fn,
    weights,
    optimizer,
    dynamical_system_normalized,
    scheduler=scheduler,
    shuffle_dataset=False,
    enable_parallel_training=False,
    update_epochs=100)
end = time.time()
time1 = end - start

################################## second training step #######################
parameters = create_parameters()

# training
optimizer = optim.Adam(model.parameters(), lr=0.5e-3)
loss_fn = nn.MSELoss()
weights = [100, 100, 100]

# training loop
start = time.time()
[model, train_loss_log1] = parallel_train(
    model,
    model1,
    training_datasets,
    parameters,
    loss_fn,
    weights,
    optimizer,
    dynamical_system_normalized,
    scheduler=None,
    shuffle_dataset=True,
    enable_parallel_training=False,
    update_epochs=50)
end = time.time()
time2 = end - start

# print training times
print(f'first training step completed in {time1} s')
print(f'second training step completed in {time2} s')

# save the model
net_state_dict = model.state_dict()
print(net_state_dict.keys())
torch.save(net_state_dict, 'model.torch')

# plot training curve
plt.figure()
plt.semilogy(range(len(train_loss_log1)),train_loss_log1)
plt.show()