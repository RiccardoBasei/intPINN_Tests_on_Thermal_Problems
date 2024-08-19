import numpy as np
import matplotlib.pyplot as plt
from create_parameters import create_parameters
from create_data import data_from_ode, power
from custom_resnet_parallel_training import CustomResNet, parallel_train
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.io import loadmat
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

# display some info
print("Info about the linear system")
print(" Number of states = " +str(np.shape(Ad)[0]))
print(" Number of inputs = " +str(np.shape(Bd)[1]-1))
print(" Number of outputs = " +str(np.shape(Cd)[0]))
num_outputs = np.shape(Cd)[0]
num_inputs = np.shape(Bd)[1]-1 # I remove ambient temp as an input
num_states = np.shape(Ad)[0]

# network definition
layers = [num_states+2, 40, 40, 40, num_states]
model = CustomResNet(layers)
torch.manual_seed(0)
print('Network initialized')

# target net definition
model1 = CustomResNet(layers)
model1.load_state_dict(model.state_dict())

########################### FIRST TRAINING STEP #########################################
# data creation
parameters = create_parameters()
parameters['intervals']['delta_t'] = (1e3,1e3)
parameters['training']['epoch_schedule'] = [2000]
parameters['training']['batching_schedule'] = [25]

# dynamical system normalized definition
xmax = torch.Tensor([parameters['intervals']['x0_1'][1],
                    parameters['intervals']['x0_2'][1],
                    parameters['intervals']['x0_3'][1]])
xmin = torch.Tensor([parameters['intervals']['x0_1'][0],
                    parameters['intervals']['x0_2'][0],
                    parameters['intervals']['x0_3'][0]])
normx = torch.diag(xmax - xmin)
dtmax = parameters['intervals']['delta_t'][1]
umax = parameters['intervals']['p'][1]

dynamical_system_normalized = {
    'A': torch.matmul(torch.Tensor(Ad), normx),
    'B': torch.Tensor(Bd)*umax,
    'E': torch.matmul(torch.Tensor(Ed), normx)/dtmax,
    'C': torch.matmul(torch.Tensor(Cd), normx),
    'D': np.matmul(torch.Tensor(Cd), xmin),
    'K': np.matmul(torch.Tensor(Ad), xmin)
    }

# dataset for the training
print('Dataset creation...')
training_dataset, steady_state_dataset = data_from_ode(parameters, dynamical_system_normalized, power)
training_datasets = {'ode_random_dataset': training_dataset,
                     'steady_state_dataset': steady_state_dataset}
print('Dataset created successfully!')

# training options
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, cooldown=25, patience=50)
loss_fn = nn.MSELoss()
weights = [0,100,0]

# training 1
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
    scheduler)
end = time.time()
time1 = end-start

###################################### SECOND TRAINING STEP ###########################################
# data creation
parameters = create_parameters()

# dynamical system normalized definition
xmax = torch.Tensor([parameters['intervals']['x0_1'][1],
                    parameters['intervals']['x0_2'][1],
                    parameters['intervals']['x0_3'][1]])
xmin = torch.Tensor([parameters['intervals']['x0_1'][0],
                    parameters['intervals']['x0_2'][0],
                    parameters['intervals']['x0_3'][0]])
normx = torch.diag(xmax - xmin)
dtmax = parameters['intervals']['delta_t'][1]
umax = parameters['intervals']['p'][1]

dynamical_system_normalized = {
    'A': torch.matmul(torch.Tensor(Ad), normx),
    'B': torch.Tensor(Bd)*umax,
    'E': torch.matmul(torch.Tensor(Ed), normx)/dtmax,
    'C': torch.matmul(torch.Tensor(Cd), normx),
    'D': np.matmul(torch.Tensor(Cd), xmin),
    'K': np.matmul(torch.Tensor(Ad), xmin)
    }

# dataset for the training
print('Dataset creation...')
training_dataset, steady_state_dataset = data_from_ode(parameters, dynamical_system_normalized, power)
training_datasets = {'ode_random_dataset': training_dataset,
                     'steady_state_dataset': steady_state_dataset}
print('Dataset created successfully!')

# training options
optimizer = optim.Adam(model.parameters(), lr=.5e-3)
loss_fn = nn.MSELoss()
weights = [130,100,0]

# training 1
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
    )
end=time.time()
time2 = end-start

# display training time information
print('')
print(f'training step 1 complented in {time1} s')
print(f'training step 2 complented in {time2} s')
print('')

# save the model
net_state_dict = model.state_dict()
print(net_state_dict.keys())
torch.save(net_state_dict, 'model1.torch')

# plot training curve
plt.figure()
plt.semilogy(range(len(train_loss_log)),train_loss_log)
plt.semilogy(range(len(train_loss_log),len(train_loss_log)+len(train_loss_log1)),train_loss_log1)
plt.show()