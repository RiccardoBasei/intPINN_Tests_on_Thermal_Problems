import numpy as np
import matplotlib.pyplot as plt
from create_parameters import create_parameters
from create_data import data_from_ode, power
from custom_resnet_parallel_training3 import CustomResNet, parallel_train
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.io import loadmat
import pandas as pd
import time

plt.close('all')

# Define the path to your .mat file
mat_file_path = 'thermal_data/python/python_data/ROM.mat'

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
layers = [num_states+2, 100, 100, 100, 100, num_states]
model = CustomResNet(layers)
torch.manual_seed(0)
print('Network initialized')

# target net definition
model1 = CustomResNet(layers)
model1.load_state_dict(model.state_dict())

# data creation
parameters = create_parameters()

# dynamical system normalized definition
maxval = []
minval = []
for var, data in parameters['intervals'].items():
    maxval.append(data[1])
    minval.append(data[0])
xmax = torch.Tensor(maxval[2:])
xmin = torch.Tensor(minval[2:])
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
while True:
    try:
       training_dataset = pd.read_csv('training_dataset.csv')
       steady_state_dataset = pd.read_csv('ss_dataset.csv')
       print('Dataset imported successfully!')
       break
    except FileNotFoundError:
        print('Dataset creation...')
        training_dataset, steady_state_dataset = data_from_ode(parameters, dynamical_system_normalized)
        print('Dataset created successfully!')
        break
training_datasets = {'steady_state_dataset': steady_state_dataset,
                     'ode_random_dataset': training_dataset}

########################### FIRST TRAINING STEP #########################################
# training options
parameters['training']['epoch_schedule'] = [720]
parameters['training']['batching_schedule'] = [100]
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, cooldown=25, patience=50)
loss_fn = nn.MSELoss()
weights = [400,100,600]

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
    scheduler=scheduler,
    shuffle_dataset=False,
    )
end = time.time()
time1 = end-start

###################################### SECOND TRAINING STEP ###########################################
# data creation
parameters = create_parameters()

# training options
optimizer = optim.Adam(model.parameters(), lr=.5e-3)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, cooldown=25, patience=50)
loss_fn = nn.MSELoss()
weights = [100,100,0]

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
    dynamical_system_normalized)
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
torch.save(net_state_dict, 'model3.torch')

# plot training curve
plt.figure(1)
plt.semilogy(range(len(train_loss_log)),train_loss_log)

plt.figure(2)
plt.semilogy(range(len(train_loss_log),len(train_loss_log)+len(train_loss_log1)),train_loss_log1)

plt.show()