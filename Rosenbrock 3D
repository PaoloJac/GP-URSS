#This case is for 3D but everything except the 'rosen' function is general and can do higher dimensions by changing a few 3's
#Preamble

import torch
import gpytorch
import math
import numpy as np
import numpy.linalg as linalg
from matplotlib import pyplot as plt

#You have to load the GP class also

#Analytic functions and training data

def F(X):
    a = 1
    b = 10
    N = X.size()[0]
    result = 0
    for i in range(N-1):
        result += b*(X[i+1]-X[i]**2)**2 + (a-X[i])**2
    return result

def J(X):
    X = torch.autograd.Variable(X, requires_grad=True)
    dtX = torch.autograd.functional.jacobian(F,X)
    return dtX

def rosen(X,Y,Z):
    a = 1
    b = 10
    f = b*(Y-X**2)**2 + (a-X)**2 + b*(Z-Y**2)**2 + (a-Y)**2
    dx = -4*b*X*(Y-X**2) - 2*(a-X)
    dy = 2*b*(Y-X**2) - 4*b*Y*(Z-Y**2) - 2*(a-Y)
    dz = 2*b*(Z-Y**2)
    return f, dx, dy, dz
    
xv, yv, zv = torch.meshgrid([torch.linspace(0, 2, 10), torch.linspace(0, 2, 10), torch.linspace(0, 2, 10)])
train_x = torch.cat((
    xv.contiguous().view(xv.numel(), 1),
    yv.contiguous().view(yv.numel(), 1),
    zv.contiguous().view(yv.numel(), 1)),
    dim=1
)

f, dfx, dfy, dfz = rosen(train_x[:, 0], train_x[:, 1], train_x[:, 2])
train_y = torch.stack([f, dfx, dfy, dfz], -1).squeeze(-1)

#First model, seeing the parameters, and then initalizing new model

likelihood, model = GP.CreateGP(train_x,train_y,dim_num=3,training_iter=800,show=True)
GP.paramshow(model)
para = (torch.tensor([1.0000e-04]),torch.tensor([0.4729, 0.4617, 1.6312]),[38.05531311035156])
likelihood, model = GP.CreateGPinitial(train_x,train_y,3,para,20,True)

#c1 plot and quasi-newton method

X = torch.tensor([0.8,0.8,0.8])
GP.c1plot(F,J,X,0.6,3)
GP.minimiseActive(F,J,X,train_x,train_y,3,0.6,0.9)
