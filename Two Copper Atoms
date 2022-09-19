#Preamble (ase is weird about installing things separately but that might just be a me problem)
import ase
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import fcc111, add_adsorbate
from ase.io import write, read
from ase.visualize import view
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.build import bulk
import torch
import gpytorch
import math
import numpy as np
import numpy.linalg as linalg
from matplotlib import pyplot as plt

#Also run the GP class

#ASE minimisation

a2 = Atoms('2Cu', positions=[(0., 0., 0.), (0., 0., 1.50)])
print(a2.positions)
a2.calc = EMT()
dyn = QuasiNewton(a2)
dyn.run(fmax=0.05)

#Creating training data

def molefunc(X,Y,Z):
    mole = Atoms('2Cu', positions=[(0., 0., 0.), (X, Y, Z)])
    mole.calc = EMT()
    f = mole.get_forces()
    dx = -max(f[:,0])
    dy = -max(f[:,1])
    dz = -max(f[:,2])
    E = mole.get_potential_energy()
    return E, dx, dy, dz

xv, yv, zv = torch.meshgrid([torch.linspace(0.1, 2, 10), torch.linspace(0.1, 2, 10), torch.linspace(0.1, 2, 10)])
train_x = torch.cat((
    xv.contiguous().view(xv.numel(), 1),
    yv.contiguous().view(yv.numel(), 1),
    zv.contiguous().view(yv.numel(), 1)),
    dim=1
)

f= []
for i,j,k in train_x:
    f.append(molefunc(i,j,k))
    
train_y = torch.tensor(f[:])
train_y = train_y.to(dtype=torch.float32)

#Model with already refined hyperparameters

params = (torch.tensor([0.0068]),torch.tensor([0.2441, 0.2475, 0.2458]),[84.41972351074219])
likelihood, model = GP.CreateGPinitial(train_x,train_y,3,params,training_iter=10,show=True)

#Functions used for minimising

def F(X):
    mole = Atoms('2Cu', positions=[(0., 0., 0.), (X[0], X[1], X[2])])
    mole.calc = EMT()
    E = mole.get_potential_energy()
    return E

def J(X):
    mole = Atoms('2Cu', positions=[(0., 0., 0.), (X[0], X[1], X[2])])
    mole.calc = EMT()
    f = mole.get_forces()
    dx = -max(f[:,0])
    dy = -max(f[:,1])
    dz = -max(f[:,2])
    return dx,dy,dz
    
#Minimising first with no active learning and then with

GP.minimisePoint(F,J,X,3,0.65,0.9,J_min=1e-5,N=100)

GP.minimiseActive(F,J,X,train_x,train_y,3,0.65,0.9,J_min=1e-5,N=100)
