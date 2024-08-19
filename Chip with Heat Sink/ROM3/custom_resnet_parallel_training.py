import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        # Convert the entire DataFrame to a tensor and store it
        self.data = torch.tensor(dataframe.iloc[:,:].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
    
class CustomResNet(nn.Module):
    
    def __init__(self, layers):
        super(CustomResNet, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # input layer
        self.input_layer = nn.Linear(in_features=layers[0], out_features=layers[1], bias=True)

        # hidden layers
        for i in range(1, len(layers)-2):
            self.hidden_layers.append(nn.Linear(in_features=layers[i], out_features=layers[i+1], bias=True))

        # output layer
        self.output_layer = nn.Linear(in_features=layers[-2], out_features=layers[-1], bias=False)

        # activation functions
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
        self.act3 = nn.Sigmoid()

        
        # dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):

        # input layer
        x = self.act2(self.input_layer(x))
        
        # hidden layers
        for self.hidden_layer in self.hidden_layers:
            x = self.act2(self.hidden_layer(x))
            x = self.dropout(x)

        # output layer
        x = self.output_layer(x)

        return x
    
def parallel_train(model, model1, training_datasets, parameters, loss_fn, w, optimizer, dynamical_system, scheduler=None, shuffle_dataset=True, enable_parallel_training=False, update_epochs=50):
    
    epochs_schedule = parameters['training']['epoch_schedule']
    batching_schedule = parameters['training']['batching_schedule']
    training_dataset = CustomDataset(training_datasets['ode_random_dataset'])
    steady_state_dataset = CustomDataset(training_datasets['steady_state_dataset'])
    
    train_loss_log = [] # list with the loss values over the epochs
    model.train() # set the model in trianing mode
    model1.eval() # set the model in evaluation mode
    
    ne = sum(epochs_schedule) # total number of epochs
    num = 0 # counter initialization
        
    for num_epochs, batch_size in zip(epochs_schedule, batching_schedule):
        train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=0)
        ss_dataloader = DataLoader(steady_state_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        for num_epoch in range(num_epochs):
            
            # update the target net and save the model
            if num_epoch % update_epochs == 0:
                print('updating target net...')
                with torch.no_grad():
                    for p1, p2 in zip(model.parameters(), model1.parameters()):
                        p2.copy_(0.5*p1 + 0.5*p2)
            
            if num_epoch % 50 == 0:
                net_state_dict = model.state_dict()
                print('saving the model...')
                name = 'intermediate_net/training_model' + str(int(num/50)) + '.torch'
                torch.save(net_state_dict, name)
                print(net_state_dict.keys())
                print('model saved.')
            
            if (num_epoch == 750 or num_epoch == 1550) and scheduler == None:
                optimizer.param_groups[0]['lr'] *= 0.2

            # number of steps per epoch
            train_loss = [] # initialize the batch loss list
            physics_loss_vec = []
            steady_state_loss_vec = []
            bounds_vec = []
            # Iterate over the dataset in batches
            for batch_data, batch_ss in zip(enumerate(train_dataloader), enumerate(ss_dataloader)):
                
                optimizer.zero_grad()

                # preprocess of the batch data (ToTensor)
                batch_data = batch_data[1].requires_grad_()
                batch_ss = batch_ss[1].requires_grad_()
                
                # preprocess steady-state dataset data and forward pass
                x_ss = model(batch_ss)
                
                # loss function
                # physical residual and bounds check
                physics_res, minmax = f(model, model1, batch_data, dynamical_system, parameters, enable_parallel_training)
                physics_loss = loss_fn(physics_res, torch.zeros(physics_res.shape))
                # steady state loss: in a ss condition, the output must be zero
                ss_loss = loss_fn(x_ss, torch.zeros(x_ss.shape))
                # weights for the loss function
                wss = w[0] # steady-state values loss
                wf = w[1] # physical residual loss
                wb = w[2] # bounds loss
                # calculate the loss
                loss = wss*ss_loss + wf*physics_loss + wb*minmax
                
                # update the weights
                loss.backward()
                
                # optimizer step
                optimizer.step(lambda: loss)

                # Save train loss for this batch
                loss_batch = loss.detach().numpy()
                train_loss.append(loss_batch)
                
                physics_loss_vec.append(wf*physics_loss.detach().numpy())
                bounds_vec.append(wb*minmax.detach().numpy())
                steady_state_loss_vec.append(wss*ss_loss.detach().numpy())

            # Save average train loss
            train_loss_avg = np.mean(train_loss)

            ss_loss_avg = np.mean(steady_state_loss_vec)
            physics_loss_avg = np.mean(physics_loss_vec)
            bounds_avg = np.mean(bounds_vec)
            print(f"EPOCH: {num_epoch+1}/{num_epochs}, BATCH SIZE: {batch_size}, AVERAGE TRAIN LOSS: {train_loss_avg}")
            train_loss_log.append(train_loss_avg)

            # scheduler step
            if scheduler != None:
                scheduler.step(train_loss_avg)
            
            print(ss_loss_avg, physics_loss_avg, bounds_avg)
            
            # counter update
            num +=1

    return model, train_loss_log

def forward_pass(model, inp):
    
    # definition of the time variable
    delta_t = inp[:,0:1]
    x0 = inp[:,2:]
    
    num_states = x0.shape[1]
    
    # network output evaluation
    x = model(inp)
    delta_t_tensor = torch.tile(delta_t,(1,num_states))
    out = x0 + torch.multiply(x, delta_t_tensor)
    
    # transpose: a column vector is needed
    out = torch.transpose(out,0,1)

    return out

def get_grad(model, inp):
    
    # definition of the time variable
    delta_t = inp[:,0:1].requires_grad_()

    # input data construction
    input_tensor = torch.hstack((delta_t, inp[:,1:]))
    
    # forward pass
    out = forward_pass(model, input_tensor)

    # derivative vector inizialization
    out_t = torch.zeros(out.shape)
    # loop all over the output variables
    for idx in range(3):
        gradient = torch.autograd.grad(
            out[idx,:],
            delta_t,
            grad_outputs = torch.ones_like(out[idx,:]),
            create_graph = True)[0].flatten()
        out_t[idx,:] = gradient
    
    return out, out_t

def bounds(out):
    
    out = torch.reshape(out,(1,-1))
    # check if the output exceeds the bound
    s_max = torch.max(torch.vstack([torch.zeros(out.shape), out - 1]),0)
    s_min = torch.min(torch.vstack([torch.zeros(out.shape), out]),0)
    
    # sum of the tensor's components
    s_max_tot = torch.sum(s_max.values)
    s_min_tot = torch.sum(s_min.values)
    
    return s_max_tot-s_min_tot

def f(model, model1, inp, dynamical_system, parameters, enable_parallel_training):

    # output of the net and its gradient
    if enable_parallel_training==False:
        out, out_t = get_grad(model, inp)
        out1 = forward_pass(model1, inp)
    else:
        out1, out_t = get_grad(model1, inp)
        out = forward_pass(model, inp)
    
    # PHYSICS EVALUATION
    # right-hand side
    Tamb_vec = 300/parameters['intervals']['p'][1]*torch.ones(inp.shape[0])
    u = torch.vstack([Tamb_vec, inp[:,1]]) # note: for this problem, the second input (Tamb = 300/norm_u) is kept constant
    K = torch.matmul(torch.reshape(dynamical_system['K'],(3,1)), torch.ones((1,inp.shape[0])))
    rhs = torch.matmul(dynamical_system['A'], out) + torch.matmul(dynamical_system['B'], u) + K
    # left-hand side
    lhs = torch.matmul(dynamical_system['E'], out_t)
    # residual evaluation
    physics_res = lhs - rhs

    # final result
    physics_res = physics_res
    
    # BOUNDS CHECK
    minmax = bounds(out)
    
    return physics_res, minmax

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True