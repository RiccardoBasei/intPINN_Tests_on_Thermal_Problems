import numpy as np
from scipy.optimize import fsolve

def normalization(sys_data, parameters):
    
    # initialization
    sys_data_norm = {
        'A': [],
        'B': [],
        'C': [],
        'D': [],
        'E': [],
        'K': []}
    
    # information about the dimension
    num_inputs = sys_data['B'].shape[1]-1 # I remove ambient temp as an input
    
    # max-min data
    maxvec = []
    minvec = []
    for var, data in parameters['intervals'].items():
        maxvec = np.hstack((maxvec, data[1]))
        minvec = np.hstack((minvec, data[0]))
    
    # normalization for input
    umax = maxvec[1:num_inputs+1]
    umin = minvec[1:num_inputs+1]
    normu = np.diag(np.hstack((300, umax-umin)))
    
    # normalization for states
    xmax = maxvec[num_inputs+1:]
    xmin = minvec[num_inputs+1:]
    normx = np.diag(xmax - xmin)
    
    # normalization for delta t
    dtmax = parameters['intervals']['delta_t'][1]
    
    # normalization
    # matrices A and K
    for Ai in sys_data['A']:
        sys_data_norm['A'].append(np.dot(Ai, normx))
        sys_data_norm['K'].append(np.dot(Ai, xmin))
    
    # matrix E
    for Ei in sys_data['E']:
        sys_data_norm['E'].append(np.dot(Ei, normx)/dtmax)
        
    # matrix B
    sys_data_norm['B'] = np.dot(sys_data['B'], normu)
    
    # matrices C and D
    sys_data_norm['C'] = np.dot(sys_data['C'], normx)
    sys_data_norm['D'] = np.dot(sys_data['C'], xmin)
        
    return sys_data_norm

def sys_norm_builder(sys, param):
    
    Atot = sys['A']
    Ktot = sys['K']
    Etot = sys['E']

    # matrix A and K
    A = Atot[0] + (Atot[1]+Atot[3]+Atot[4]+Atot[5]+Atot[6])*param[0] + Atot[2]*param[1]
    K = Ktot[0] + (Ktot[1]+Ktot[3]+Ktot[4]+Ktot[5]+Ktot[6])*param[0] + Ktot[2]*param[1]

    # matrix E
    E = Etot[0] + (Etot[1]+Etot[2]+Etot[3]+Etot[4]+Etot[5])*param[2]

    # matrix B
    B = sys['B']

    # matrix C and D
    C = sys['C']
    D = sys['D']

    # dynamical system in dict form
    dyn_system = {'A': A, 'E': E, 'B': B, 'C': C, 'D': D, 'K': K}

    return dyn_system

def ode_solver_nl(sys, sys_builder, power, t_vec, p_vec, param, t_span, x0, dt):
    
    # initialization
    xold = x0
    y0 = np.dot(sys['C'], x0) + sys['D']
    pp = param(y0)
    xvec = x0
    yvec = y0
    tvec = np.arange(t_span[0], t_span[1], dt)
    
    # for cycle over time
    for tt in tvec:
        
        #print(f'solving: {int(tt/t_span[1]*10000)/100} %')
        
        # input evaluation
        power_now = power(tt*1e0, t_vec, p_vec)/120
        u = np.hstack([1,power_now,power_now,power_now,power_now])
        
        # dynamical system at instant t
        sys = sys_builder(pp)
        
        # discrete system generation
        Ed = sys['E'] - sys['A']*dt
        Ad = sys['E']
        Bd = sys['B']*dt
        try:
            Kd = sys['K']*dt
        except KeyError:
            Kd = 0
        try:
            Dd = sys['D']
        except KeyError:
             Dd = 0
            
        # next step evaluation
        rhs = np.dot(Ad,xold) + np.dot(Bd,u).flatten() + Kd
        xnew = np.linalg.solve(Ed, rhs)
        
        # output evaluation
        ynew = np.dot(sys['C'], xnew) + sys['D']
        
        # updates
        pp = param(ynew)
        xold = xnew
        
        # vector concatenation
        xvec = np.vstack([xvec,xnew])
        yvec = np.vstack([yvec,ynew])
        
    return xvec[:-1,:], yvec[:-1,:], tvec

def ss_evaluation(sys, sys_builder, param, p_vec):
        
    num_states = sys['B'].shape[0]
    xss = np.zeros(num_states)
    for pi in p_vec:
        
        # input evaluation        
        u = np.hstack([1, pi, pi, pi ,pi])
        
        # dynamical system
        Temp = lambda x: np.dot(sys['C'], x) + sys['D']
        pp = lambda Temp: param(Temp)
        A = lambda pp: sys_builder(pp)['A']   
        B = lambda pp: sys_builder(pp)['B']        
        K = lambda pp: sys_builder(pp)['K']        
        
        # steady state evaluation
        F = lambda x: np.dot(A(pp(Temp(x))), x) + np.dot(B(pp(Temp(x))), u) + K(pp(Temp(x)))
        xssi = fsolve(F, np.zeros((num_states,1)))
        
        # store the result in the final vector
        xss = np.vstack([xss, xssi])
        
    return xss[1:,:]