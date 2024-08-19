import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

        # dropout defintion
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
    
def parallel_train(model, model1, training_datasets, parameters, loss_fn, w, optimizer, scheduler, sys_info, shuffle_dataset=True, update_epochs=50):
    
    dynamical_system = sys_info['sys_matrices']
    mat_par = sys_info['param']
    
    epochs_schedule = parameters['training']['epoch_schedule']
    batching_schedule = parameters['training']['batching_schedule']
    training_dataset = CustomDataset(training_datasets['ode_random_dataset'])
    steady_state_dataset = CustomDataset(training_datasets['steady_state_dataset'])
    
    train_loss_log = [] # list with the loss values over the epochs
    model.train() # set the model in evaluation mode

    ne = sum(epochs_schedule) # total number of epochs
    num = 0 # counter initialization
    
    for num_epochs, batch_size in zip(epochs_schedule, batching_schedule):
        train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=0)
        ss_dataloader = DataLoader(steady_state_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        sys_tensor = tensor_building(dynamical_system, batch_size)
        
        train_loss = [] # initialize the batch loss list
        for num_epoch in range(num_epochs):
            
            # update the target net
            if num_epoch % update_epochs == 0:
                print('updating target net...')
                with torch.no_grad():
                    for p1, p2 in zip(model.parameters(), model1.parameters()):
                        p2.copy_(1*p1 + 0*p2)
            
            # save the model
            if num_epoch % 20 == 0:
                net_state_dict = model.state_dict()
                print('saving the model...')
                name = 'intermediate_net/training_model' + str(int(num/20)) + '.torch'
                torch.save(net_state_dict, name)
                print(net_state_dict.keys())
                print('model saved.')

            # Iterate over the dataset in batches
            train_loss = []
            physics_train_loss = []
            bounds_train_loss = []
            ss_train_loss = []
            for batch_data, batch_ss in zip(enumerate(train_dataloader), enumerate(ss_dataloader)):
                
                # preprocess of the batch data (ToTensor)
                batch_data = batch_data[1].requires_grad_()
                batch_ss = batch_ss[1].requires_grad_()
                
                # preprocess steady-state dataset data and forward pass
                x_ss = model(batch_ss)
                                        
                # set the gradient to zero
                optimizer.zero_grad()
                
                # loss function
                # physical residual and bounds check
                physics_res, minmax = f(model, model1, batch_data, sys_tensor, batch_size, mat_par)
                physics_loss = loss_fn(physics_res, torch.zeros(physics_res.shape))
                # steady state loss: in a ss condition, the output must be zero
                ss_loss = loss_fn(x_ss, torch.zeros(x_ss.shape))
                # weights for the loss function: w_n = (w_fin/w_in)^(n/n_tot)*w_in
                wss = w[0] # ss loss
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
                
                physics_train_loss.append(wf*physics_loss.detach().numpy())
                bounds_train_loss.append(wb*minmax.detach().numpy())
                ss_train_loss.append(wss*ss_loss.detach().numpy())

            # Save average train loss
            train_loss_avg = np.mean(train_loss)
            print(f"EPOCH: {num_epoch+1}/{num_epochs}, BATCH SIZE: {batch_size}, AVERAGE TRAIN LOSS: {train_loss_avg}")
            train_loss_log.append(train_loss_avg)

            # scheduler step
            scheduler.step(train_loss_avg)
            #print(np.mean(ss_train_loss), np.mean(physics_train_loss), np.mean(bounds_train_loss))
            #print(optimizer.param_groups[-1]['lr'])     
            
            # counter update
            num +=1

    return model, train_loss_log

def forward_pass(model, inp):

    # definition of the time variable
    delta_t = inp[:,0:1]
    x0 = inp[:,2:]
    
    num_states = x0.shape[1]

    # network output evaluation
    delta_t_tensor = torch.tile(delta_t, (1,num_states))
    x = model(inp)
    out = x0 + torch.multiply(x, delta_t_tensor)
    
    # transpose: a column vector is needed
    out = torch.transpose(out,0,1)

    return out

def get_grad(model, inp):
    
    # definition of the time variable
    delta_t = inp[:,0:1].requires_grad_()

    # input data construction
    input_tensor = torch.hstack((delta_t, inp[:,1:]))
    num_states = input_tensor.shape[1]-2
    
    # forward pass
    out = forward_pass(model, input_tensor)

    # derivative vector inizialization
    out_t = torch.zeros(out.shape)
    # loop all over the output variables
    for idx in range(num_states):
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
    s_max = torch.max(torch.vstack([torch.zeros(out.shape), out-1]),0)
    s_min = torch.min(torch.vstack([torch.zeros(out.shape), out]),0)
    
    # sum of the tensor's components
    s_max_tot = torch.sum(s_max.values)
    s_min_tot = torch.sum(s_min.values)
    
    return s_max_tot-s_min_tot

def f(model, model1, inp, sys_tensor, n_points, param):
    
    # output of the net and its gradient
    out, out_t = get_grad(model, inp)
    out1 = forward_pass(model1, inp)
    
    # physical residual evaluation
    physics_res = physical_residual_evaluation(sys_tensor, param, inp, out, out1, out_t, n_points)
    
    # bounds check
    minmax = bounds(out)
    
    return physics_res, minmax

def tensor_building(rom_norm, n_points):
        
    # matrix conversion to tensors
    B = torch.Tensor(rom_norm['B'])
    C = torch.Tensor(rom_norm['C'])
    D = torch.Tensor(rom_norm['D'])
    
    A0 = torch.Tensor(rom_norm['A'][0])
    A1 = torch.Tensor(rom_norm['A'][1]+rom_norm['A'][3]+rom_norm['A'][4]+rom_norm['A'][5]+rom_norm['A'][6])
    A2 = torch.Tensor(rom_norm['A'][2])
    
    K0 = torch.Tensor(rom_norm['K'][0])
    K1 = torch.Tensor(rom_norm['K'][1]+rom_norm['K'][3]+rom_norm['K'][4]+rom_norm['K'][5]+rom_norm['K'][6])
    K2 = torch.Tensor(rom_norm['K'][2])
    
    E0 = torch.Tensor(rom_norm['E'][0])
    E1 = torch.Tensor(rom_norm['E'][1]+rom_norm['E'][2]+rom_norm['E'][3]+rom_norm['E'][4]+rom_norm['E'][5])
    
    # matrix preparation for matrix multiplication
    A0_expanded = A0.unsqueeze(2)
    A1_expanded = A1.unsqueeze(2)
    A2_expanded = A2.unsqueeze(2)
    
    B_expanded = B.unsqueeze(2)
    
    K0_expanded = K0.unsqueeze(1)
    K1_expanded = K1.unsqueeze(1)
    K2_expanded = K2.unsqueeze(1)
    
    E0_expanded = E0.unsqueeze(2)
    E1_expanded = E1.unsqueeze(2)
    
    # save tensors in a dict
    sys_tensor = {'A':[A0_expanded,A1_expanded,A2_expanded],
                  'K':[K0_expanded,K1_expanded,K2_expanded],
                  'E':[E0_expanded,E1_expanded],
                  'B':B_expanded,
                  'C':C,
                  'D':D}
    
    return sys_tensor
    
def physical_residual_evaluation(sys_tensor, param, inp, x, x1, dx, n_points):
    
    num_states = sys_tensor['B'].shape[0]
    
    # unwrap input dict
    A0_expanded = sys_tensor['A'][0]
    A1_expanded = sys_tensor['A'][1]
    A2_expanded = sys_tensor['A'][2]
    
    K0_expanded = sys_tensor['K'][0]
    K1_expanded = sys_tensor['K'][1]
    K2_expanded = sys_tensor['K'][2]

    E0_expanded = sys_tensor['E'][0]
    E1_expanded = sys_tensor['E'][1]
        
    B_expanded = sys_tensor['B']
    C = sys_tensor['C']
    D = sys_tensor['D']
    
    # temperature evaluation from states
    T = torch.matmul(C, x) + torch.transpose(torch.tile(D, (n_points,1)),0,1)
    
    # parameters value at temperature T
    try:
        param_val = param(T)
    except RuntimeError:
        param_val = param(T.detach().numpy())
    k_Si = torch.Tensor(param_val[0])
    k_cer = torch.Tensor(param_val[1])
    cp_Si = torch.Tensor(param_val[2])
    
    # input tensor creation
    Tamb_vec = torch.ones(inp.shape[0])
    u = torch.vstack([Tamb_vec, inp[:,1], inp[:,1], inp[:,1], inp[:,1]]) # note: for this problem, the second input (Tamb = 300/norm_u) is kept constant
    
    # parameters dimension preparation
    k_Si  = k_Si.view(1,1,n_points)
    k_cer = k_cer.view(1,1,n_points)
    cp_Si = cp_Si.view(1,1,n_points)
    ones = torch.ones((1,1,n_points))
    
    # matrix tensors evaluation
    A0t = torch.transpose(torch.transpose(A0_expanded*ones,2,1),1,0)
    A1t = torch.transpose(torch.transpose(A1_expanded*k_Si,2,1),1,0)
    A2t = torch.transpose(torch.transpose(A2_expanded*k_cer,2,1),1,0)
    A = A0t + A1t + A2t
    
    Bt = torch.transpose(torch.transpose(B_expanded*ones,2,1),1,0)
    
    K0t = torch.transpose(torch.transpose(K0_expanded*ones,2,1),1,0)
    K1t = torch.transpose(torch.transpose(K1_expanded*k_Si,2,1),1,0)
    K2t = torch.transpose(torch.transpose(K2_expanded*k_cer,2,1),1,0)
    K = torch.reshape(K0t + K1t + K2t, (n_points,num_states,1))
    
    E0t = torch.transpose(torch.transpose(E0_expanded*ones,2,1),1,0)    
    E1t = torch.transpose(torch.transpose(E1_expanded*cp_Si,2,1),1,0)
    E = E0t + E1t
    
    # states and derivative preparation
    xt  = torch.transpose(x.unsqueeze(2),1,0)
    dxt = torch.transpose(dx.unsqueeze(2),1,0)
    ut  = torch.transpose(u.unsqueeze(2),1,0)
    
    # matrix vector multiplication
    lhs = torch.matmul(E, dxt)
    rhs = torch.matmul(A, xt) + K + torch.matmul(Bt, ut)
    res = rhs - lhs
    
    return res

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True