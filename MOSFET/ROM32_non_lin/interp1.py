import torch

def interp1(xnodes,ynodes,xval):

    xnodes = torch.hstack([torch.Tensor([0]),xnodes,xnodes[-1]*1e17])
    ynodes = torch.hstack([ynodes[0],ynodes,ynodes[-1]])

    yval = torch.Tensor([])
    for xi in xval:
        index=0
        for x in xnodes:
            if xi>x:
                index+=1

            yi = (ynodes[index] - ynodes[index-1])/(xnodes[index] - xnodes[index-1])*(xi-xnodes[index]) + ynodes[index]
        
        yval = torch.hstack([yval,yi])


    return yval