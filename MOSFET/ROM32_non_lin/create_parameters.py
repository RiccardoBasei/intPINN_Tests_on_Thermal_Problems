from scipy.io import loadmat
import numpy as np

def create_parameters():

       # load the .mat file
       min_max_states = loadmat('non_lin_data/limits.mat', mat_dtype=True)['limits']

       intervals = {
              'delta_t': np.array((1e-2,1e0)),
              'p': np.array((0,120)),
              'p2': np.array((0,120)),
              'p3': np.array((0,120)),
              'p4': np.array((0,120))}
       for i in range(0,min_max_states.shape[0]):
              var = 'x0_' + str(i+1)
              intervals[var] = min_max_states[i]

       training = {
              'n_training_points': 0, # this is the number of random points of the training dataset, not the whole number
              'epoch_schedule': [1000],
              'batching_schedule': [50]
              }
    
       parameters = {
              'intervals': intervals,
              'training': training
              }

       return parameters