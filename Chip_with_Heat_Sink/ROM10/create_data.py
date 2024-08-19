import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def power(t, t_vec, p_vec):
    
    index = 0
    for tlim in t_vec:
        if t < tlim:
            index +=1
            
    p = p_vec[-index]
    
    return p

def data_from_ode(parameters, dynamical_system_normalized):
    
    #num_outputs = np.shape(dynamical_system_normalized['C'])[0]
    num_states = np.shape(dynamical_system_normalized['A'])[0]
    
    # input signal variables and delta_t values
    delta_t = 1/parameters['intervals']['delta_t'][1]*np.logspace(
        np.log10(parameters['intervals']['delta_t'][0]),
        np.log10(parameters['intervals']['delta_t'][1]),
        num=50)
    p = 1/parameters['intervals']['p'][1]*np.linspace(
        parameters['intervals']['p'][0],
        parameters['intervals']['p'][1],
        num=10)

    # tensor to array conversion
    dyn_sys_numpy = {}
    for var, data in dynamical_system_normalized.items():
        dyn_sys_numpy[var] = data.detach().numpy()
    
    ######### steady state-dataset #########
    # initialization
    xss = np.zeros((1, num_states))
    data_ss = {}
    # steady state evaluation
    for pi in p:
        u = np.vstack((300/parameters['intervals']['p'][1], pi)) # input vector
        rhs = np.matmul(dyn_sys_numpy['B'], u).T + dyn_sys_numpy['K'] # rhs vector
        xssi = np.linalg.solve(dyn_sys_numpy['A'], -rhs.T) # steady state condition
        
        xss = np.vstack([xss, xssi.T[0,:]])
    xss = xss[1:,:] # first row removed
    
    # delta t vector for steady-state
    delta_t_ss = 1/parameters['intervals']['delta_t'][1]*np.logspace(
        np.log10(parameters['intervals']['delta_t'][0]),
        np.log10(parameters['intervals']['delta_t'][1]),
        num=70) 
    
    # definition of the vector containing all the data to build the dict
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
    t_vec = [0.5e4,1e4,1.5e4,2e4,2.5e4,3e4,3.5e4,4e4,4.5e4,5e4,5.5e4,6e4]#,6.5e4,7e4]
    p_vec = [0,1,9,4,7,2,0,8,3,5,6,8]#,3,1]
    
    # dynamical system normalized definition
    dtmax = parameters['intervals']['delta_t'][1]
    umax = parameters['intervals']['p'][1]
        
    # dynamical system model preparation for the ode-solver
    mat = np.linalg.inv(dyn_sys_numpy['E'])
    mat1 = np.matmul(mat, dyn_sys_numpy['A'])
    mat2 = np.matmul(mat, dyn_sys_numpy['B'])
    
    # input vector
    u = lambda t: np.array([300/umax, power(t*dtmax, t_vec, p_vec)/umax]).T
    
    # ode function: dx/dt = F(x,t)
    F = lambda t, x: np.matmul(mat1, x) + np.matmul(mat2, u(t)) + np.matmul(mat, dyn_sys_numpy['K'])
    
    # time interval
    t_span = (0.0, np.max(t_vec)/dtmax)
    
    # initial value
    x0 = np.linalg.solve(dyn_sys_numpy['A'], -np.dot(dyn_sys_numpy['B'], np.array([300/9,0])) - dyn_sys_numpy['K'])
    
    # ode resolution
    sol = solve_ivp(F, t_span, x0, t_eval=None)
    x = sol.y
    y = np.dot(dyn_sys_numpy['C'], x) +  np.tile(dyn_sys_numpy['D'], (sol.t.shape[0],1)).T
    
    # plot the training datatset
    plt.figure(100)
    plt.plot(sol.t*dtmax,y[0,:])
    plt.plot(sol.t*dtmax,y[1,:])
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [°C]')
    plt.legend(['Chip','Heat Sink'])
    plt.grid(linewidth=0.5)

    plt.figure(101)
    tplot = np.hstack([0,np.repeat(t_vec[:-1],2),t_vec[-1]])
    pplot = np.repeat(p_vec,2)
    plt.plot(tplot,pplot)
    plt.xlabel('Time [s]')
    plt.ylabel('Power [W]')
    plt.grid(linewidth=0.5)
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
    #df_ode = pd.concat([df_ode, df_ode_i])
    
    ################# final concatenation ###########################
    df = pd.concat([df_ss2, df_ode, df_random], ignore_index=True)

    # save the dataset as csv file
    df.to_csv('training_dataset.csv',index=False)
    df_ss.to_csv('ss_dataset.csv',index=False)

    return df, df_ss