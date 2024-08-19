def create_parameters():

       intervals = {
              'delta_t': (1e1, 1e3),
              'p': (0, 9),
              'x0_1': (-2.2319597e4, -1.4945458e4),
              'x0_2': (-46.961671, 113.187051),
              'x0_3': (-5.98404, 10.482705)
              }
       
       training = {
              'n_training_points': 0, # this is the number of random points of the training dataset, not the whole number
              'epoch_schedule':    [3000],
              'batching_schedule': [200]
              }
    
       parameters = {
              'intervals': intervals,
              'training': training
              }

       return parameters