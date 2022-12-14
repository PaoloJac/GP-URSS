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

#class with all major/general functions

class GP():
    def mean_f(X):
        return likelihood(model(X)).mean
    
    def P(Z,dim_num):
        X = Z.reshape(1,dim_num)
        tX = torch.autograd.Variable(X, requires_grad=True)
        dtX = torch.autograd.functional.jacobian(GP.mean_f, tX)
        dtX = np.delete(dtX,0,1)
        dtX = dtX.reshape(dim_num,dim_num)
        return np.matrix(dtX)

    def CreateGP(train_x,train_y,dim_num,training_iter=200,show=False):
        class GPModelWithDerivatives(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMeanGrad()
                self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=dim_num)
                self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=dim_num+1)
        model = GPModelWithDerivatives(train_x, train_y, likelihood)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05) 
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if show is True:
                if i == 0:
                    print("Iter %d/%d - Loss: %.3f   Noise: %.3f" % (
                        i + 1, training_iter, loss.item(),
                        model.likelihood.noise.item()
                    ), end = '   ')
                else:
                    print("\nIter %d/%d - Loss: %.3f   Noise: %.3f" % (
                        i + 1, training_iter, loss.item(),
                        model.likelihood.noise.item()
                    ), end = '   ')
            optimizer.step()
        model.eval()
        likelihood.eval()
        return likelihood,model
    
    def CreateGPinitialRaw(train_x,train_y,dim_num,params,training_iter=200,show=False):
        class GPModelWithDerivatives(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMeanGrad()
                self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=dim_num)
                self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=dim_num+1)
        model = GPModelWithDerivatives(train_x, train_y, likelihood)
        model.likelihood.initialize(raw_task_noises=torch.tensor([params[0]]))
        model.likelihood.initialize(raw_noise=torch.tensor([params[1]]))
        model.mean_module.initialize(constant=torch.tensor(params[2]))
        model.base_kernel.initialize(raw_lengthscale=torch.tensor([[params[3]]]))
        model.covar_module.initialize(raw_outputscale=torch.tensor([params[4]]))
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05) 
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if show is True:
                if i == 0:
                    print("Iter %d/%d - Loss: %.3f   Noise: %.3f" % (
                        i + 1, training_iter, loss.item(),
                        model.likelihood.noise.item()
                    ), end = '   ')
                else:
                    print("\nIter %d/%d - Loss: %.3f   Noise: %.3f" % (
                        i + 1, training_iter, loss.item(),
                        model.likelihood.noise.item()
                    ), end = '   ')
                print('Lengthscales:', end = ' ')
                for n in range(dim_num):
                    print('%.3f' % model.covar_module.base_kernel.lengthscale.squeeze()[n], end= ' ')
            optimizer.step()
        model.eval()
        likelihood.eval()
        return likelihood,mode
    
    def CreateGPinitial(train_x,train_y,dim_num,params,training_iter=200,show=False):
        class GPModelWithDerivatives(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMeanGrad()
                self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=dim_num)
                self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=dim_num+1)
        model = GPModelWithDerivatives(train_x, train_y, likelihood)
        model.likelihood.noise = params[0]
        model.covar_module.base_kernel.lengthscale = params[1]
        model.covar_module.outputscale = params[2]
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05) 
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if show is True:
                if i == 0:
                    print("Iter %d/%d - Loss: %.3f   Noise: %.3f" % (
                        i + 1, training_iter, loss.item(),
                        model.likelihood.noise.item()
                    ), end = '   ')
                else:
                    print("\nIter %d/%d - Loss: %.3f   Noise: %.3f" % (
                        i + 1, training_iter, loss.item(),
                        model.likelihood.noise.item()
                    ), end = '   ')
                #print('Lengthscales:', end = ' ')
                #for n in range(dim_num):
                #    print('%.3f' % model.covar_module.base_kernel.lengthscale.squeeze()[n], end= ' ')
            optimizer.step()
        model.eval()
        likelihood.eval()
        return likelihood,model
        
    def minimisePoint(F,J,X,dim_num,c1,rho,J_min=1e-6,N=100):
        for i in range(N):
            func = F(X)
            grad = J(X)
            if linalg.norm(grad)<J_min:
                break
            pk = linalg.solve(GP.P(X,dim_num),np.negative(grad))
            a0 = 1 #max step value
            while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                    a0 = a0*rho
            X = X + a0*pk
            X = X.to(dtype=torch.float32)
            print(i+1, '/', N)
            print(X,F(X),linalg.norm(J(X)),a0)

    def minimiseArray(F,J,X,dim_num,c1,rho,J_min=1e-6,N=100):
        for n in X:
            for i in range(N):
                func = F(n)
                grad = J(n)
                if linalg.norm(grad)<J_min:
                    print(n,i+1)
                    break
                pk = linalg.solve(GP.P(n,dim_num),np.negative(grad))
                a0=1
                while not F(n+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                    a0 = a0*rho
                n = n + a0*pk
                #n = n.to(dtype=torch.float32)
                if i is N-1:
                    print('Fail',n)
                    break
        
    def minimiseActive(F,J,X,train_x,train_y,dim_num,c1,rho,J_min=1e-6,N=100):
        for i in range(N):
            func = F(X)
            grad = J(X)
            if linalg.norm(grad)<J_min:
                break
            pk = -linalg.solve(GP.P(X,dim_num),np.negative(grad))
            a0 = 1
            while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                if a0 > 0.05: #np.dot(J(X+a0*pk),pk) < 0.9*np.dot(grad,pk): 
                    a0 = a0*rho
                else:
                    break
            X = X + a0*pk
            X = X.to(dtype=torch.float32)
            print(i+1, '/', N)
            print(func,linalg.norm(grad),a0)
            x = torch.reshape(X,(1,dim_num))
            train_x = torch.cat([train_x,x])
            y = torch.cat([torch.tensor(F(X)).unsqueeze(-1),torch.tensor(J(X))],0)                           
            train_y = torch.cat([train_y,y.unsqueeze(-1).T])
            train_y = train_y.to(dtype=torch.float32)
            model.set_train_data(train_x,train_y,strict=False)
            
    #This function does a spectral decom of the hessian before each step. I haven't tested this fully so there may be some bugs
    def minimiseSpecActive(F,J,X,train_x,train_y,dim_num,c1,rho,J_min=1e-6,N=100):
        I = np.identity(57)
        for i in range(N):
            func = F(X)
            grad = J(X)
            if linalg.norm(grad)<J_min:
                break
            P = GP.P(X,dim_num)
            valuesraw = np.linalg.eig(P)[0]
            values = np.zeros((57,57))
            for s in range(57):
                values[s,s] += valuesraw[s]
            vectors = np.linalg.eig(P)[1]
            vectorinverse = linalg.solve(vectors,I)
            P2 = np.dot(abs(values),vectorinverse)
            P2 = np.dot(vectors,P2)
            pk = -linalg.solve(P2,np.negative(grad))
            a0 = 1
            while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
                if a0 > 0.05: #np.dot(J(X+a0*pk),pk) < 0.9*np.dot(grad,pk): 
                    a0 = a0*rho
                else:
                    break
            X = X + a0*pk
            X = X.to(dtype=torch.float32)
            print(i+1, '/', N)
            print(func,linalg.norm(grad),a0)
            x = torch.reshape(X,(1,dim_num))
            train_x = torch.cat([train_x,x])
            y = torch.cat([torch.tensor(F(X)).unsqueeze(-1),torch.tensor(J(X))],0)                           
            train_y = torch.cat([train_y,y.unsqueeze(-1).T])
            train_y = train_y.to(dtype=torch.float32)
            model.set_train_data(train_x,train_y,strict=False)
            
    def c1plot(F,J,X,c1,dim_num):
        pk = linalg.solve(GP.P(X,dim_num),np.negative(J(X)))
        alpha = torch.linspace(0,1,100)
        alpha = alpha.tolist()
        instep = []
        outstep = []
        for i in range(100):
            instep.append(F(X+(alpha[i])*pk))
            outstep.append(F(X) + c1*alpha[i]*np.dot(J(X),pk))
        plt.plot(instep, 'r',outstep, 'b')
        
    def paramshow(model, raw=False):
        if raw is True:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
        if raw is False:
            print(f'Noise: {model.likelihood.noise.data}')
            print(f'Lengthscale: {model.covar_module.base_kernel.lengthscale.data}')
            print(f'Outputscale: {model.covar_module.outputscale.data}')
