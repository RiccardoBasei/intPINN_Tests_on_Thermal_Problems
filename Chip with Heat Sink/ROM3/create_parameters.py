def create_parameters():

       intervals = {
              'delta_t': (1e1, 1e3),
              'p': (0, 9),
              'x0_1': (-1.949e4, -1.49e4),
              'x0_2': (4.93, 122.7),
              'x0_3': (0.332, 10.465)
              }
       
       training = {
              'n_training_points': 0, # this is the number of random points of the training dataset, not the whole number
              'epoch_schedule': [5000],
              'batching_schedule': [25],
              }
       
       parameters = {
              'intervals': intervals,
              'training': training,
              }

       return parameters