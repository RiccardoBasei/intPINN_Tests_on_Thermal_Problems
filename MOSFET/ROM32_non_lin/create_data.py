import numpy as np
import pandas as pd
from data_import import sys_norm_builder, ode_solver_nl, ss_evaluation
import matplotlib.pyplot as plt
from scipy.io import savemat

def power(t, t_vec, p_vec):
    
    index = 0
    for tlim in t_vec:
        if t < tlim:
            index +=1
            
    p = p_vec[-index]
    
    return p

def data_from_ode(parameters, rom_norm, param):
    
    #num_outputs = np.shape(rom_norm['C'])[0]
    num_inputs = np.shape(rom_norm['B'])[1]-1 # I remove ambient temp as an input
    num_states = np.shape(rom_norm['B'])[0]
    
    # input signal variables and delta_t values
    delta_t = 1/parameters['intervals']['delta_t'][1]*np.logspace(
        np.log10(parameters['intervals']['delta_t'][0]),
        np.log10(parameters['intervals']['delta_t'][1]),
        num=50)
    p = 1/parameters['intervals']['p'][1]*np.linspace(
        parameters['intervals']['p'][0],
        parameters['intervals']['p'][1],
        num=13)
    
    ######### steady state-dataset #########
    # initialization

    # steady states evaluation    
    xss = ss_evaluation(rom_norm, lambda par: sys_norm_builder(rom_norm,par), param, p)
    
    # delta t vec for ss dataset
    delta_t_ss = 1/parameters['intervals']['delta_t'][1]*np.logspace(
        np.log10(parameters['intervals']['delta_t'][0]),
        np.log10(parameters['intervals']['delta_t'][1]),
        num=50)
    
    # definition of the vector containing all the data to build the dict
    row1 = np.repeat(delta_t_ss, p.size)
    row2 = np.tile(p, delta_t_ss.size)
    row345 = np.tile(xss.T, delta_t_ss.size)

    data_vec = np.vstack([row1, row2, row345])
        
    # steady-state dictionary definition
    data_ss = {}
    i = 0
    for var, interval in parameters['intervals'].items():
        if var != 'p2' and var != 'p3' and var != 'p4':
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
        if var != 'p2' and var != 'p3' and var != 'p4':
            data[var] = data_vec[i,:]
            i +=1
    
    # create a DataFrame from the steady-state data
    df_ss2 = pd.DataFrame(data)
    
    ##################### random dataset ####################à
    data_random = {}
    # Generate random data within the specified intervals
    for var, interval in parameters['intervals'].items():
        if var != 'p2' and var != 'p3' and var != 'p4':
           data_random[var] = np.random.uniform(0, 1, parameters['training']['n_training_points'])
    
    # overwrite the first rows of the dict: we are interested only in some values
    data_random['delta_t'] = np.repeat(delta_t, int(parameters['training']['n_training_points']/delta_t.size))
    data_random['p'] = np.tile(p, int(parameters['training']['n_training_points']/p.size))
    
    # definition of a DataFrame object
    df_random = pd.DataFrame(data_random)
    
    ############## ode solver dataset ############
    # power info
    t_vec = [6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96]
    p_vec = [70,10,50,100,120,0,70,60,40,110,20,30,100,120,80,0]
    
    # normalization factors definition
    maxvec = []
    minvec = []
    for var, data in parameters['intervals'].items():
        maxvec = np.hstack((maxvec, data[1]))
        minvec = np.hstack((minvec, data[0]))
    umax = maxvec[1:num_inputs+1]
    umin = minvec[1:num_inputs+1]
    normu = np.diag(np.hstack((300, umax - umin)))
    denormu = np.linalg.inv(normu)
    dtmax = parameters['intervals']['delta_t'][1]
    
    # time interval
    t_span = (0.0, max(t_vec)/dtmax)
    
    # training dataset evaluation
    x0 = xss[0,:]
    sol_x, sol_y, sol_t = ode_solver_nl(rom_norm, lambda par: sys_norm_builder(rom_norm,par),
                                        power, t_vec, p_vec, param, t_span, x0, .5e-2/dtmax)
    sol_x = sol_x.T

    # training dataset visualization
    plt.figure(100)
    plt.plot(sol_t,sol_y[:,0])
    plt.plot(sol_t,sol_y[:,1])
    plt.plot(sol_t,sol_y[:,2])
    plt.plot(sol_t,sol_y[:,3])
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid(linewidth=.5)
    
    # vector of data
    p = [0]
    for t in sol_t:
        p = np.hstack((p, power(t*dtmax, t_vec, p_vec)/120))
    data_vec = np.vstack((sol_t, p[:-1], sol_x[:,:])).T
    np.random.shuffle(data_vec)
        
    # from vector to dict
    data_ode = {}
    i = 0
    for var, interval in parameters['intervals'].items():
        if var != 'p2' and var != 'p3' and var != 'p4':
            data_ode[var] = data_vec[:,i]
            i +=1

    # overwrite the delta_t row with the values of interest
    data_ode['delta_t'] = np.repeat(delta_t, int(sol_t.size/delta_t.size))
    
    # make all the rows of the same length
    n_samples = (data_ode['delta_t'].size // 1000) * 1000
    for var, interval in parameters['intervals'].items():
        if var != 'p2' and var != 'p3' and var != 'p4':
            data_ode[var] = data_ode[var][:n_samples]
        
    # definition of a DataFrame object
    df_ode = pd.DataFrame(data_ode)
    #df_ode = pd.concat([df_ode, df_ode_i])
    
    ################# final concatenation ###########################
    df = pd.concat([df_ss2, df_ode, df_random], ignore_index=True)
    
    return df, df_ss