import numpy as np
import matplotlib.pyplot as plt
from create_parameters import create_parameters
from custom_resnet_parallel_training import CustomResNet
import torch
from scipy.io import loadmat
from data_import import normalization, sys_norm_builder, ode_solver_nl
import pandas as pd
import time

def power(t, t_vec, p_vec):
    
    index = 0
    for tlim in t_vec:
        if (t - tlim) >= -1e-4*t:
            index +=1
            
    p = p_vec[index]
    
    return p

def evaluation(model, t0, x0, delta_t_vec, p_steps, t_steps, parameters, dyn_sys):
        
    dtmax = parameters['intervals']['delta_t'][1]
    umax = parameters['intervals']['p'][1]
    
    # normalize input data
    x0 = torch.Tensor(x0)
    C = torch.Tensor(dyn_sys['C'])
    D = torch.Tensor(dyn_sys['D'])
    delta_t_vec_norm = delta_t_vec/dtmax
    y0 = torch.matmul(C, x0) + D
    
    # initialization
    xvec = x0.detach().numpy()
    yvec = y0.detach().numpy()
    tvec = [t0]
    
    t = t0
    xold = x0
    for dt in delta_t_vec_norm.T:
        
        # power at time t
        p = power(t, t_steps, p_steps)

        # create input tensor
        dt = torch.Tensor([dt])/dtmax
        p = torch.Tensor([p])
        inp = torch.hstack((dt, p/umax, xold))
        
        # following step evaluation
        xnew = xold + model(inp)*dt

        # output variables evaluation
        ynew = torch.matmul(C, xnew) + D
        
        # vectors update
        xvec = np.vstack((xvec, xnew.detach().numpy()))
        yvec = np.vstack((yvec, ynew.detach().numpy()))
        
        # update initial step state and time value
        xold = xnew
        
        # time and time vector update
        t = t + dt.detach().numpy()*dtmax
        tvec = np.hstack((tvec, t))
        
    return xvec, yvec, tvec

def RKevaluation(filename, t_eval):

    data = loadmat(filename)
    t = data['RK1']['t'][0][0][0]
    y = data['RK1']['y'][0][0]
    y_eval = np.zeros([y.shape[0], t_eval.shape[0]])
    for i in range(y.shape[0]):
        y_eval[i,:] = np.interp(t_eval, t, y[i,:])

    return y_eval.T

plt.close('all')

# import material lookup tables
k_Si_data = pd.read_csv('non_lin_data/LUT_variables/k_Si_fun1.csv')
k_cer_data = pd.read_csv('non_lin_data/LUT_variables/k_ceramic_fun1.csv')
cp_Si_data = pd.read_csv('non_lin_data/LUT_variables/cp_Si_fun1.csv')
mat_data = [k_Si_data, k_cer_data, cp_Si_data]

# interpolation of data
k_Si  = lambda T_Si: np.interp(T_Si, k_Si_data['T'].values, k_Si_data['k_Si'].values)
k_cer = lambda T_cer: np.interp(T_cer, k_cer_data['T'].values, k_cer_data['k_cer'].values)
cp_Si = lambda T_Si: np.interp(T_Si, cp_Si_data['T'].values, cp_Si_data['cp_Si'].values)

param_fT = lambda T: [k_Si(T[3]), k_cer(T[2]), cp_Si(T[3])]

# # plot the material characteristics
# plt.figure(1)
# Temp = np.linspace(200,800,150)
# plt.plot(Temp,k_Si(Temp))
# plt.plot(k_Si_data['T'], k_Si_data['k_Si'],'.')
# plt.figure(2)
# Temp = np.linspace(200,800,150)
# plt.plot(Temp,k_cer(Temp))
# plt.plot(k_cer_data['T'], k_cer_data['k_cer'],'.')
# plt.figure(3)
# Temp = np.linspace(200,1900,150)
# plt.plot(Temp,cp_Si(Temp))
# plt.plot(cp_Si_data['T'], cp_Si_data['cp_Si'],'.')

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
dtmax = parameters['intervals']['delta_t'][1]

# info about the dynamical system
num_outputs = np.shape(rom_norm['C'])[0]
num_inputs = np.shape(rom_norm['B'])[1]-1 # I remove ambient temp as an input
num_states = np.shape(rom_norm['A'][0])[0]

# network definition
model_dict = torch.load('model.torch')
del model_dict['hidden_layer.weight']
del model_dict['hidden_layer.bias']
layers = [num_states+2, 200, 200, 200, 200, 200, num_states]
model = CustomResNet(layers)
model.load_state_dict(model_dict)

# set the evaluation mode
model.eval()

# intial condition
Tamb_vec = 300*np.ones(4)
p0 = param_fT(Tamb_vec)
rom_4ic = sys_norm_builder(rom_norm, p0)
u0 = np.array([1,0,0,0,0])
x0_ic = np.linalg.solve(rom_4ic['A'], -np.dot(rom_4ic['B'], u0) - rom_4ic['K'])

# first dataset
x0 = torch.Tensor(x0_ic)
t_vec = [12,24,36,48,60,72,84,96,100]
p_vec = [0,50,100,120,100,20,90,60,0]

# delta t vector creation
delta_t = 0.5e0
delta_t_vec = delta_t*np.ones(int(max(t_vec)/delta_t))

# PINN evaluation
t_start = time.time()
xvec, yvec, tvec = evaluation(model, 0, x0, delta_t_vec*dtmax, p_vec, t_vec, parameters, rom_norm)
t_end = time.time()
tPINN = t_end - t_start
    
# euler evaluation
t_start = time.time()
sol_x, sol_y, sol_t = ode_solver_nl(rom_norm, lambda par: sys_norm_builder(rom_norm, par),
                                    power, t_vec, p_vec, param_fT,
                                    (0.0, 100), x0_ic, delta_t)
t_end = time.time()
teul = t_end - t_start

# RK evaluation
yRK = RKevaluation('RK_results1.mat', tvec)

# data for plots
data = loadmat('RK_results1.mat')
tplot = data['RK1']['t'][0][0][0]
yplot = data['RK1']['y'][0][0]

# PINN Error estimation
PINN_error = np.abs(yvec - yRK)
PINN_error_rel = np.divide(np.abs(yvec - yRK), yRK)

# Euler error estimation
try:
    eul_error = np.abs(sol_y - yRK[:-1,:])
    eul_error_rel = np.divide(np.abs(sol_y - yRK[:-1,:]), yRK[:-1,:])
except ValueError:
    eul_error = np.abs(sol_y - yRK)
    eul_error_rel = np.divide(np.abs(sol_y - yRK), yRK)

# plot results
plt.figure(1)
for i in range(yvec.shape[1]):
    plt.plot(tplot, yplot[i,:],'b')  
    plt.plot(tvec,yvec[:,i],'r--')
    plt.grid(linewidth=.5)
    plt.legend(['RK','PINN'])
    plt.ylabel('Temperature [K]')
    plt.xlabel('Time [s]')


# display results
print('')
print('first dataset')
print(f'evaluation times: Euler: {teul}, PINN: {tPINN}')
print('')
print('PINN')
print(f'abs:  max error: {np.max(PINN_error)}°C, MS error: {np.sqrt(np.mean(PINN_error**2))}°C')
print(f'rel:  max error: {np.max(PINN_error_rel)*100}%, MS error: {np.sqrt(np.mean(PINN_error_rel**2))*100}%')
print('')
print('Euler')
print(f'abs:  max error: {np.max(eul_error)}°C, MS error: {np.sqrt(np.mean(eul_error**2))}°C')
print(f'rel:  max error: {np.max(eul_error_rel)*100}%, MS error: {np.sqrt(np.mean(eul_error_rel**2))*100}%')


plt.show()