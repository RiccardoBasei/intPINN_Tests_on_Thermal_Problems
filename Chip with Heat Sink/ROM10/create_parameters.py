from scipy.io import loadmat
import numpy as np

def create_parameters():

       # load the .mat file
       m = loadmat('thermal_data/limits.mat', mat_dtype=True)['limits']

       intervals = {
              'delta_t': np.array((1e1,1e3)),
              'p': np.array((0,9))}
       for i in range(0,m.shape[0]):
              var = 'x0_' + str(i+1)
              intervals[var] = [m[i,0]-0.0*m[i,0]*np.sign(m[i,0]), m[i,1]+0.0*m[i,1]*np.sign(m[i,1])]

       training = {
              'n_training_points': 0, # this is the number of random points of the training dataset, not the whole number
              'epoch_schedule': [3000],
              'batching_schedule': [50]
              }
    
       parameters = {
              'intervals': intervals,
              'training': training
              }

       return parameters