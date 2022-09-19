#Preamble
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

#Also need to load GP class

#ASE minimisation

a1 = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
del a1[[0,0,0]]
a1.calc = EMT()
dyn = QuasiNewton(a1)
dyn.run(fmax=0.09)

#Making the training data

def trainfunc(X):
    slab = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
    del slab[[0,0,0]]
    x = X[::3].reshape(19,1)
    y = X[1::3].reshape(19,1)
    z = X[2::3].reshape(19,1)
    Y = torch.cat([x,y,z],1)
    slab.positions = Y
    slab.calc = EMT()
    E = slab.get_potential_energy()
    E = torch.tensor([E])
    f = slab.get_forces().tolist()
    d = torch.tensor(np.negative(f),dtype=torch.float32)
    d = d.reshape(57)
    result = torch.cat([E,d])
    return result
    
slab = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
del slab[[0,0,0]]
X = torch.tensor(slab.positions)
X = X.to(dtype=torch.float32)
X = X.reshape(57)

train_x = torch.tensor([])
for i in range(150):
    train_x = torch.cat([train_x,X + 0.1*np.random.normal(0,3,57)])
train_x = train_x.reshape(150,57).to(dtype=torch.float32)

f = []
for i in train_x:
    f.append(trainfunc(i))
    
train_y = torch.stack(f)
train_y = train_y.to(dtype=torch.float32)

#Making numerical hessian for starting atom positions

slab = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
del slab[[0,0,0]]
slab.calc = EMT()
h = 1e-5
X0 = slab.get_positions()
H = np.zeros([3*len(slab),3*len(slab)])

for n in range(len(slab)):
    for i in range(3):
        slab.positions[n,i] += h
        fplus = slab.get_forces()
        slab.positions[n,i] -= 2*h
        fminus = slab.get_forces()
        slab.positions = X0
        H[3*n+i,:] = -(fplus.flatten()-fminus.flatten())/(2*h)
        
#Minimisation functions

def F(X):
    slab = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
    del slab[[0,0,0]]
    x = X[::3].reshape(19,1)
    y = X[1::3].reshape(19,1)
    z = X[2::3].reshape(19,1)
    Y = torch.cat([x,y,z],1)
    slab.positions = Y
    slab.calc = EMT()
    E = slab.get_potential_energy()
    return E

def J(X):
    slab = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
    del slab[[0,0,0]]
    x = X[::3].reshape(19,1)
    y = X[1::3].reshape(19,1)
    z = X[2::3].reshape(19,1)
    Y = torch.cat([x,y,z],1)
    slab.positions = Y
    slab.calc = EMT()
    f = slab.get_forces().tolist()
    d = torch.tensor(np.negative(f),dtype=torch.float32)
    d = d.reshape(57)
    return d
    
#Loading initial Hessian and doing a spectral decomposition

P = GP.P(X,57)
valuesraw = np.linalg.eig(P)[0]
values = np.zeros((57,57))
for i in range(57):
    values[i,i] += valuesraw[i]
I = np.identity(57)
vectors = np.linalg.eig(P)[1]
vectorinverse = linalg.solve(vectors,I)
P2 = np.dot(abs(values),vectorinverse)
P2 = np.dot(vectors,P2)

#c1 plots with spectral decom hessian and then identity matrix

pk = linalg.solve(P2,np.negative(J(X)))
c1 = 0.005
alpha = torch.linspace(0,1,100)
alpha = alpha.tolist()
instep = []
outstep = []
for i in range(100):
    instep.append(F(X+(alpha[i])*pk))
    outstep.append(F(X) + c1*alpha[i]*np.dot(J(X),pk))
plt.plot(instep, 'r',outstep, 'b')

I = np.identity(57)
pk = linalg.solve(I,np.negative(J(X)))
c1 = 0.65
alpha = torch.linspace(0,1,100)
alpha = alpha.tolist()
instep = []
outstep = []
for i in range(100):
    instep.append(F(X+(alpha[i])*pk))
    outstep.append(F(X) + c1*alpha[i]*np.dot(J(X),pk))
plt.plot(instep, 'r',outstep, 'b')

#Attempted minimisations (they don't work, they step in the wrong direction or over the minimium)

GP.minimiseActive(F,J,X,train_x,train_y,57,0.4,0.9,J_min=1e-5,N=100)
GP.minimiseSpecActive(F,J,X,train_x,train_y,57,0.2,0.9,J_min=1e-5,N=100)

#General functions for BFGS minimisations, first using an inputed starting hessian and then with an identiy matrix initial matrix

def minimisePointBFGS(F,J,X,P,c1,rho,J_min=1e-6,N=100):
    for i in range(N):
        func = F(X)
        grad = J(X)
        if linalg.norm(grad)<J_min:
            break
        pk = linalg.solve(P,np.negative(grad))
        a0 = 1 
        while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                a0 = a0*rho
        sk = a0*pk
        X = X + sk
        X = X.to(dtype=torch.float32)
        yk = J(X)-grad
        P = P + (np.dot(yk,yk)/np.dot(yk,sk)) - (np.dot(np.dot(P,sk),np.dot(sk,P))/np.dot(sk,np.dot(P,sk)))
        print(i+1, '/', N)
        print(F(X),linalg.norm(J(X)),a0)
        
def minimisePointIdentity(F,J,X,c1,rho,J_min=1e-6,N=100):
    I = np.identity(57)
    for i in range(N):
        func = F(X)
        grad = J(X)
        if linalg.norm(grad)<J_min:
            break
        pk = linalg.solve(I,np.negative(grad))
        a0 = 1 #max step value
        while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                a0 = a0*rho
        X = X + a0*pk
        X = X.to(dtype=torch.float32)
        print(i+1, '/', N)
        print(F(X),linalg.norm(J(X)),a0)
        
#I also wrote some functions using the numerical hessian. I don't think they worked but I've but them here in case they might be useful. 
#The numerical hessians formed are specific to this atom body

def minimisePointNum(F,J,X,c1,rho,J_min=1e-6,N=100):
    h = 1e-5
    H = np.zeros([3*19,3*19])
    for s in range(N):
        func = F(X)
        grad = J(X)
        if linalg.norm(grad)<J_min:
            break
        slab = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
        del slab[[0,0,0]]
        x = X[::3].reshape(19,1)
        y = X[1::3].reshape(19,1)
        z = X[2::3].reshape(19,1)
        Y = torch.cat([x,y,z],1)
        slab.positions = Y
        slab.calc = EMT()
        for n in range(len(slab)):
            for i in range(3):
                slab.positions[n,i] += h
                fplus = slab.get_forces()
                slab.positions[n,i] -= 2*h
                fminus = slab.get_forces()
                slab.positions = Y
                H[3*n+i,:] = -(fplus.flatten()-fminus.flatten())/(2*h)
        pk = linalg.solve(H,np.negative(grad))
        a0 = 1
        while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                    a0 = a0*rho
        X = X + a0*pk
        X = X.to(dtype=torch.float32)
        print(s+1, '/', N)
        print(F(X),linalg.norm(J(X)),a0)
        
def minimisePointNumSpec(F,J,X,c1,rho,J_min=1e-6,N=100):
    h = 1e-5
    H = np.zeros([3*19,3*19])
    for s in range(N):
        func = F(X)
        grad = J(X)
        if linalg.norm(grad)<J_min:
            break
        slab = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
        del slab[[0,0,0]]
        x = X[::3].reshape(19,1)
        y = X[1::3].reshape(19,1)
        z = X[2::3].reshape(19,1)
        Y = torch.cat([x,y,z],1)
        slab.positions = Y
        slab.calc = EMT()
        for n in range(len(slab)):
            for i in range(3):
                slab.positions[n,i] += h
                fplus = slab.get_forces()
                slab.positions[n,i] -= 2*h
                fminus = slab.get_forces()
                slab.positions = Y
                H[3*n+i,:] = -(fplus.flatten()-fminus.flatten())/(2*h)
        valuesraw = np.linalg.eig(H)[0]
        values = np.zeros((57,57))
        for s in range(57):
            values[s,s] += valuesraw[s]
        vectors = np.linalg.eig(H)[1]
        vectorinverse = linalg.solve(vectors,I)
        H = np.dot(abs(values),vectorinverse)
        H = np.dot(vectors,H)
        pk = linalg.solve(H,np.negative(grad))
        a0 = 1
        while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                    a0 = a0*rho
        X = X + a0*pk
        X = X.to(dtype=torch.float32)
        print(s+1, '/', N)
        print(F(X),linalg.norm(J(X)),a0)
        
def minimisePointNumSpecBFGS(F,J,X,c1,rho,J_min=1e-6,N=100):
    h = 1e-5
    slab = ase.build.fcc100('Cu',(5,2,2), a=4., vacuum = 5)
    del slab[[0,0,0]]
    x = X[::3].reshape(19,1)
    y = X[1::3].reshape(19,1)
    z = X[2::3].reshape(19,1)
    Y = torch.cat([x,y,z],1)
    slab.positions = Y
    slab.calc = EMT()
    H = np.zeros([3*len(slab),3*len(slab)])
    for n in range(len(slab)):
        for i in range(3):
            slab.positions[n,i] += h
            fplus = slab.get_forces()
            slab.positions[n,i] -= 2*h
            fminus = slab.get_forces()
            slab.positions = Y
            H[3*n+i,:] = -(fplus.flatten()-fminus.flatten())/(2*h)
    valuesraw = np.linalg.eig(H)[0]
    values = np.zeros((57,57))
    for s in range(57):
        values[s,s] += valuesraw[s]
    vectors = np.linalg.eig(H)[1]
    vectorinverse = linalg.solve(vectors,I)
    H = np.dot(abs(values),vectorinverse)
    H = np.dot(vectors,H)
    for i in range(N):
        func = F(X)
        grad = J(X)
        if linalg.norm(grad)<J_min:
            break
        pk = linalg.solve(H,np.negative(grad))
        a0 = 1 
        while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                a0 = a0*rho
        sk = a0*pk
        X = X + sk
        X = X.to(dtype=torch.float32)
        yk = J(X)-grad
        H = H + (np.dot(yk,yk)/np.dot(yk,sk)) - np.dot(np.dot(H,sk),np.dot(sk,H))/np.dot(sk,np.dot(H,sk))
        print(i+1, '/', N)
        print(F(X),linalg.norm(J(X)),a0)
