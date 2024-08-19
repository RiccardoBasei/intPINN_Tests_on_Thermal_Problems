import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import torch

def power(t, t_vec, p_vec):
    
    index = 0
    for tlim in t_vec:
        if t < tlim:
            index +=1
            
    p = p_vec[-index]
    
    return p

def data_from_ode(parameters, dynamical_system_normalized, power):
    
    #num_outputs = np.shape(dynamical_system_normalized['C'])[0]
    num_states = np.shape(dynamical_system_normalized['A'])[0]
    
    # input signal variables and delta_t values
    delta_t = 1/parameters['intervals']['delta_t'][1]*np.logspace(
        np.log10(parameters['intervals']['delta_t'][0]),
        np.log10(parameters['intervals']['delta_t'][1]),
        num=30)
    p = 1/parameters['intervals']['p'][1]*np.linspace(
        parameters['intervals']['p'][0],
        parameters['intervals']['p'][1],
        num=10)

    # tensor to array conversion
    dyn_sys_numpy = {}
    for var, data in dynamical_system_normalized.items():
        try:
            dyn_sys_numpy[var] = data.detach().numpy()
        except AttributeError:
            dyn_sys_numpy[var] = data
    
    ######### steady state-dataset #########
    # initialization
    xss = np.zeros((1, num_states))
    data_ss = {}
    # steady state evaluation
    for pi in p:
        T = lambda x: (np.dot(dyn_sys_numpy['C'], x) + dyn_sys_numpy['D'])[0]
        u = lambda T: np.vstack((300/parameters['intervals']['p'][1], dyn_sys_numpy['coeff'](T)*pi)) # input vector
        rhs = lambda T: np.dot(dyn_sys_numpy['B'], u(T)).T + dyn_sys_numpy['K'] # rhs vector
        res = lambda x: np.dot(dyn_sys_numpy['A'], x) + rhs(T(x)).flatten() # physical residual evaluation
        xssi = fsolve(res, np.zeros(num_states)) # steady state condition evaluation
        # vector update
        xss = np.vstack([xss, xssi])
    xss = xss[1:,:] # first row removed
    
    # definition of the vector containing all the data to build the dict
    delta_t_ss = 1/parameters['intervals']['delta_t'][1]*np.logspace(
        np.log10(parameters['intervals']['delta_t'][0]),
        np.log10(parameters['intervals']['delta_t'][1]),
        num=50) 
    row1 = np.repeat(delta_t_ss, p.size)
    row2 = np.tile(p, delta_t_ss.size)
    row345 = np.tile(xss.T, delta_t_ss.size)

    data_vec = np.vstack([row1, row2, row345])
        
    # steady-state dictionary definition
    i = 0
    for var, interval in parameters['intervals'].items():
        data_ss[var] = data_vec[i,:]
        i +=1
        
    # create a DataFrame from the steady-state data
    df_ss = pd.DataFrame(data_ss)
        
    # definition of the vector containing all the data to build the dict
    row1 = np.repeat(np.repeat(delta_t, p.size), p.size)
    row2 = np.tile(np.repeat(p, p.size), delta_t.size)
    row345 = np.tile(np.tile(xss.T, p.size), delta_t.size)

    data_vec = np.vstack([row1, row2, row345])
    
    data = {}
    # steady-state dictionary definition
    i = 0
    for var, interval in parameters['intervals'].items():
        data[var] = data_vec[i,:]
        i +=1
    
    # create a DataFrame from the steady-state data
    df_ss2 = pd.DataFrame(data)
    
    ##################### random dataset ####################à
    data_random = {}
    # Generate random data within the specified intervals
    for var, interval in parameters['intervals'].items():
        data_random[var] = np.random.uniform(0, 1, parameters['training']['n_training_points'])
    
    # overwrite the first rows of the dict: we are interested only in some values
    data_random['delta_t'] = np.repeat(delta_t, int(parameters['training']['n_training_points']/delta_t.size))
    data_random['p'] = np.tile(p, int(parameters['training']['n_training_points']/p.size))
    
    # definition of a DataFrame object
    df_random = pd.DataFrame(data_random)
    
    ############## ode solver dataset ############
    # power info
    t_vec = [1e4,2e4,3e4,4e4,5e4,6e4,7e4,8e4,9e4,10e4,11e4,12e4,13e4,14e4,15e4]#,16e4,17e4,18e4,19e4,20e4]
    p_vec = [0,1,9,4,7,2,0,6,3,5,2,8,9,7,3]#,4,0,1,2,4]
    
    # dynamical system normalized definition
    dtmax = parameters['intervals']['delta_t'][1]
    umax = parameters['intervals']['p'][1]
        
    # dynamical system model preparation for the ode-solver
    mat = lambda T: np.linalg.inv(dyn_sys_numpy['E']*dyn_sys_numpy['f'](T))
    mat1 = lambda T: np.matmul(mat(T), dyn_sys_numpy['A'])
    mat2 = lambda T: np.matmul(mat(T), dyn_sys_numpy['B'])
    
    # temperature evaluation
    T = lambda x: (np.dot(dyn_sys_numpy['C'], x) + dyn_sys_numpy['D'])[0]

    # input vector
    u = lambda T, t: np.array([300/umax, dyn_sys_numpy['coeff'](T)*power(t*dtmax, t_vec, p_vec)/umax]).T
    
    # ode function: dx/dt = F(x,t)
    F = lambda t, x: np.matmul(mat1(T(x)), x) + np.matmul(mat2(T(x)), u(T(x),t)) + np.matmul(mat(T(x)), dyn_sys_numpy['K'])
    
    # time interval
    t_span = (0.0, np.max(t_vec)/dtmax)
    
    # initial value
    x0 = np.linalg.solve(dyn_sys_numpy['A'], -np.dot(dyn_sys_numpy['B'], u(300,0)) - dyn_sys_numpy['K'])
    
    # ode resolution
    sol = solve_ivp(F, t_span, x0, t_eval=None)
    x = sol.y
    y = np.dot(dyn_sys_numpy['C'], x) + np.tile(dyn_sys_numpy['D'], (sol.t.shape[0],1)).T
    
    # plot the training datatset
    plt.figure(100)
    plt.plot(sol.t*dtmax,y[0,:])
    plt.plot(sol.t*dtmax,y[1,:])
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [°C]')
    plt.grid(linewidth=.5)
    
    plt.figure(101)
    tplot = np.hstack([0,np.repeat(t_vec[:-1],2),t_vec[-1]])
    pplot = np.repeat(p_vec,2)
    plt.plot(tplot,pplot)
    plt.xlabel('Time [s]')
    plt.ylabel('Power [W]')
    plt.grid(linewidth=.5)
    plt.show()

    # delta t evaluation starting from delta t
    delta_t_solver = sol.t[1:] - sol.t[:-1]
    
    # vector of data
    p = [0]
    for t in sol.t:
        p = np.hstack((p, power(t*dtmax, t_vec, p_vec)/umax))
    data_vec = np.vstack((delta_t_solver, p[1:-1], x[:,:-1])).T
    np.random.shuffle(data_vec)
        
    # from vector to dict
    data_ode = {}
    i = 0
    for var, interval in parameters['intervals'].items():
        data_ode[var] = data_vec[:,i]
        i +=1

    # overwrite the delta_t row with the values of interest
    data_ode['delta_t'] = np.repeat(delta_t, int(sol.t.size/delta_t.size))
    
    # make all the rows of the same length
    n_samples = (data_ode['delta_t'].size // 1000) * 1000
    for var, interval in parameters['intervals'].items():
        data_ode[var] = data_ode[var][:n_samples]
        
    # definition of a DataFrame object
    df_ode = pd.DataFrame(data_ode)
    
    ################# final concatenation ###########################
    df = pd.concat([df_ss2, df_ode, df_random], ignore_index=True)

    return df, df_ss
