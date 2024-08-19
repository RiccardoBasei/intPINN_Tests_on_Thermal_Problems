import torch
from custom_resnet_parallel_training import CustomResNet
import matplotlib.pyplot as plt
import numpy as np
from create_parameters import create_parameters
from scipy.io import loadmat
from scipy.integrate import solve_ivp
import time

def power(t, t_vec, p_vec):
    index = 0
    for tlim in t_vec:
        if t < tlim:
            index +=1
            
    p = p_vec[-index]
    return p

def evaluation(model, t0, x0, delta_t_vec, p_steps, t_steps, parameters, dynamical_system):
        
    # normalization factors
    max = []
    min = []
    for var, data in parameters['intervals'].items():
        max.append(data[1])
        min.append(data[0])
    xmax = torch.Tensor(max[2:])
    xmin = torch.Tensor(min[2:])
    normx = torch.diag(xmax - xmin)
    denormx = torch.linalg.inv(normx)
    dtmax = parameters['intervals']['delta_t'][1]
    umax = parameters['intervals']['p'][1]
        
    # normalize input data
    x0 = torch.Tensor(x0)
    x0_norm = torch.matmul(x0-xmin, denormx)
    delta_t_vec_norm = delta_t_vec/dtmax
    C = torch.matmul(torch.Tensor(dynamical_system['C']), normx)
    D = torch.matmul(torch.Tensor(dynamical_system['C']), xmin)
    y0 = torch.matmul(C, x0_norm) + D
    
    # initialization
    xvec = x0_norm.detach().numpy()
    yvec = y0.detach().numpy()
    tvec = [t0]
    
    t = t0
    xold = x0_norm
    for dt in delta_t_vec_norm.T:
        
        # power at time t
        p = power(t, t_steps, p_steps)
        
        # create input tensor
        dt = torch.Tensor([dt])
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
        t = t + dt*dtmax
        tvec = np.hstack((tvec, t.detach().numpy()))
        
    return xvec, yvec, tvec

def RKevaluation(dynamical_system, t_vec, p_vec, x0, t_eval=None):
    
    t_vec = np.array(t_vec)
    p_vec = np.array(p_vec)
    x0 = np.array(x0)
    
    # dynamical system model preparation for the ode-solver
    mat = np.linalg.inv(dynamical_system['E'])
    mat1 = np.matmul(mat, dynamical_system['A'])
    mat2 = np.matmul(mat, dynamical_system['B'])
    
    # input vector
    u = lambda t: np.array([300, power(t, t_vec, p_vec)])
    
    # ode function: dx/dt = F(x,t)
    F = lambda t, x: np.matmul(mat1, x) + np.matmul(mat2, u(t))
    
    # time interval
    t_span = (0.0, np.max(t_vec))
    
    # ode resolution
    sol = solve_ivp(F, t_span, x0, t_eval=t_eval)
    x = sol.y
    
    # output evaluation
    y = np.matmul(dynamical_system['C'], x)
    
    return sol.t, y

def Euler_evaluation(dynamical_system, p_vec, t_vec, t0, x0, delta_t):
    
    # discretized system
    Ad = dynamical_system['E']
    Bd = delta_t*dynamical_system['B']
    Cd = dynamical_system['C']
    Ed = dynamical_system['E'] - delta_t*dynamical_system['A']
    
    xold = np.array(x0)
    t = np.arange(t0, max(t_vec), delta_t)
    y = np.dot(Cd, x0)
    yvec = y
    for tt in t:
        
        # input evaluation
        u = np.array([300, power(tt, t_vec, p_vec)])
        
        # next step evaluation
        mat = np.dot(Ad, xold) + np.dot(Bd, u)
        xnew = np.linalg.solve(Ed, mat)
        
        # output evaluation
        ynew = np.dot(Cd, xnew)
        
        # state variable update
        xold = xnew
        
        # update output vector
        yvec = np.vstack((yvec, ynew))
        
    return t, yvec
        
        
# Define the path to your .mat file
mat_file_path = 'thermal_data/python/python_data/matlab_data.mat'

# Load the .mat file
data = loadmat(mat_file_path, mat_dtype=True)
rom = data['rom']

# linear system
dynamical_system = {
    'A': rom['A'][0][0],
    'B': rom['B'][0][0],
    'C': rom['C'][0][0],
    'E': rom['E'][0][0]
    }

# info about the linear system
num_outputs = np.shape(dynamical_system['C'])[0]
num_inputs = np.shape(dynamical_system['B'])[1]-1 # I remove ambient temp as an input
num_states = np.shape(dynamical_system['A'])[0]

# data creation
parameters = create_parameters()

# network definition
model_dict = torch.load('model.torch')
del model_dict['hidden_layer.weight']
del model_dict['hidden_layer.bias']
layers = [num_states+2, 40, 40, 40, num_states]
model = CustomResNet(layers)
model.load_state_dict(model_dict)

# set the evaluation mode
model.eval()

################################## first dataset ##############################

# initial conditions
x0 = np.linalg.solve(dynamical_system['A'], -np.dot(dynamical_system['B'], np.array([300,0]).T))

# input power
p_steps = [0, 1, 9, 4, 7, 2, 0]
t_steps = [1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4]

# delta t
delta_t = 0.5e3
delta_t_vec = np.ones(int(max(t_steps)/delta_t))*delta_t

## EVALUATION
# PINN evaluation
start_time = time.time()  
xvec, yvec, tvec = evaluation(model, 0.0, x0, delta_t_vec, p_steps, t_steps, parameters, dynamical_system)
end_time = time.time()
tPINN = end_time - start_time

# RK evaluation
start_time = time.time()  
t, y = RKevaluation(dynamical_system, t_steps, p_steps, x0, t_eval=tvec)
end_time = time.time()
tRK = end_time - start_time

# Euler evaluation
start_time = time.time()  
tvec_eul, yeul = Euler_evaluation(dynamical_system, p_steps, t_steps, 0.0, x0, delta_t)
end_time = time.time()
teul = end_time - start_time

# plot the results
plt.figure(1)
for idx in range(2):
    plt.plot(t, y[idx,:])
    plt.plot(tvec, yvec[:,idx],'--')
    #plt.plot(tvec_eul, yeul[:-1,idx])
plt.legend(['RK chip','PINN chip','RK sink','PINN sink'])
plt.xlabel('Time [s]')
plt.ylabel('Temperature [°C]')
plt.grid(linewidth=.5)

# error evaluation
PINN_error = np.abs(yvec - y.T)
eul_error = np.abs(yeul[:,:] - y.T)

# display errors
print('')
print('first dataset')
print(f'evaluation times: RK: {tRK}, Euler: {teul}, PINN: {tPINN}')
print('')
print(f'PINN:  max error: {np.max(PINN_error)}°C, mean error: {np.sqrt(np.mean(PINN_error**2))}°C')
print(f'Euler: max error: {np.max(eul_error)}°C, mean error: {np.sqrt(np.mean(eul_error**2))}°C')

############################## second dataset #################################

# initial conditions
x0 = np.linalg.solve(dynamical_system['A'], -np.dot(dynamical_system['B'], np.array([300,0]).T))

# input power
p_steps = [0, 1.25, 3.5, 4.175, 8.5, 6.4, 1.5]
t_steps = [0.6e4, 1.2e4, 1.8e4, 2.4e4, 3.0e4, 3.6e4, 4.2e4]

# time step
delta_t = 2.5e2
delta_t_vec = np.ones(int(max(t_steps)/delta_t))*delta_t

## EVALUATION
# PINN evaluation
start_time = time.time()  
xvec, yvec, tvec = evaluation(model, 0.0, x0, delta_t_vec, p_steps, t_steps, parameters, dynamical_system)
end_time = time.time()
tPINN = end_time - start_time

# RK evaluation
start_time = time.time()  
t, y = RKevaluation(dynamical_system, t_steps, p_steps, x0, t_eval=tvec)
end_time = time.time()
tRK = end_time - start_time

# Euler evaluation
start_time = time.time()  
tvec_eul, yeul = Euler_evaluation(dynamical_system, p_steps, t_steps, 0.0, x0, delta_t)
end_time = time.time()
teul = end_time - start_time

# plot the results
plt.figure(2)
for idx in range(2):
    plt.plot(t, y[idx,:])
    plt.plot(tvec, yvec[:,idx],'--')
    #plt.plot(tvec_eul, yeul[:-1,idx])
plt.legend(['RK chip','PINN chip','RK sink','PINN sink'])
plt.xlabel('Time [s]')
plt.ylabel('Temperature [°C]')
plt.grid(linewidth=.5)

# error evaluation
PINN_error = np.abs(yvec - y.T)
eul_error = np.abs(yeul[:,:] - y.T)

# display errors
print('')
print('second dataset')
print(f'evaluation times: RK: {tRK}, Euler: {teul}, PINN: {tPINN}')
print('')
print(f'PINN:  max error: {np.max(PINN_error)}°C, MS error: {np.sqrt(np.mean(PINN_error**2))}°C')
print(f'Euler: max error: {np.max(eul_error)}°C, MS error: {np.sqrt(np.mean(eul_error**2))}°C')

# show the plots
plt.show()