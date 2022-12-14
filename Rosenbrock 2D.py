#Preamble
import torch
import math
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as linalg

%matplotlib inline
%load_ext autoreload
%autoreload 2

#Analytical Stuff

def RB(X,Y):
    a = 1.
    b = 100.
    f = (a-X)**2+b*(Y-X**2)**2
    return f

def RBJac(X,Y):
    a = 1.
    b = 100.
    dx = -2*(a-X)-4*b*X*(Y-X**2)
    dy = 2*b*(Y-X**2)
    return dx, dy

def RBHess(X,Y):
    a = 1.
    b = 100.
    dxx = 2-4*b*Y+12*b*X**2
    dxy = -4*b*X
    dyx = -4*b*X
    dyy = 2*b
    return dxx,dxy,dyx,dyy
    
#Plots

fig, ax = plt.subplots(1, 3, figsize=(14, 10))
n1, n2 = 50, 100
xv, yv = torch.meshgrid([torch.linspace(-2, 2, n1), torch.linspace(-1, 3, n2)])
f = RB(xv,yv)
fx, fy = RBJac(xv,yv)
extent = (xv.min(), xv.max(),yv.min(),yv.max())
p = ax[0].imshow(np.log(f).T, extent=extent, cmap=cm.jet_r, origin='lower')
ax[0].set_title('f')
#plt.colorbar(p)
#ax[0].plot
ax[1].imshow(fx.T, extent=extent, cmap=cm.jet,origin='lower')
ax[1].set_title('fx')
ax[2].imshow(fy.T, extent=extent, cmap=cm.jet,origin='upper')
ax[2].set_title('fy')

#Function Re-written for Minimising (You want to functions to only have one vector input rather than multiple paramters otherwise things get awkward)

def F(X):
    a = 1
    b = 100
    return (a-X[0])**2+b*(X[1]-X[0]**2)**2

def J(X):
    a = 1
    b = 100
    dx = -2*(a-X[0])-4*b*X[0]*(X[1]-X[0]**2)
    dy = 2*b*(X[1]-X[0]**2)
    return dx, dy

def H(X):
    a = 1
    b = 100
    dxx = 2-4*b*X[1]+12*b*X[0]**2
    dxy = -4*b*X[0]
    dyx = -4*b*X[0]
    dyy = 2*b
    return np.matrix([[dxx,dxy],[dyx,dyy]])
    
#Newton Method with No Linesearch

n = 100
X = [-10.,10.]
J_min = 1e-8

for i in range(n):
    X = X + linalg.solve(H(X),np.negative(J(X)))
    print(i+1, '/', n)
    print(X,F(X),linalg.norm(J(X)))
    if linalg.norm(J(X))<J_min:
        break
        
#Newton Method with LineSearch

N = 100
X = [-10,10.]
c1 = 0.3
rho = 0.9
for i in range(N):
    func = F(X)
    grad = J(X)
    if linalg.norm(grad)<J_min:
        break
    pk = linalg.solve(H(X),np.negative(grad))
    a0 = 1
    while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
            a0 = a0*rho
    X = X + a0*pk
    print(i+1, '/', N)
    print(X,F(X),linalg.norm(J(X)),a0)
    
#GP Model Stuff

def RB(X,Y):
    a = 1
    b = 10
    f = (a-X)**2+b*(Y-X**2)**2
    dx = -2*(a-X)-4*b*X*(Y-X**2)
    dy = 2*b*(Y-X**2)
    return f, dx ,dy
   
def F(X):
    a = 1
    b = 10
    return (a-X[0])**2+b*(X[1]-X[0]**2)**2

def J(X):
    a = 1
    b = 10
    dx = -2*(a-X[0])-4*b*X[0]*(X[1]-X[0]**2)
    dy = 2*b*(X[1]-X[0]**2)
    return dx, dy
 
#Making the Training Data

xv, yv = torch.meshgrid([torch.linspace(0, 1, 30), torch.linspace(0, 1, 30)])
train_x = torch.cat((
    xv.contiguous().view(xv.numel(), 1),
    yv.contiguous().view(yv.numel(), 1)),
    dim=1
)

f, dfx, dfy = RB(train_x[:, 0], train_x[:, 1])
train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)

train_y += 0.05 * torch.randn(train_y.size())

#Making the Model

class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
model = GPModelWithDerivatives(train_x, train_y, likelihood)

#Seeing and changing the parameters (I made functions that do this for you, see class)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
        
model.likelihood.initialize(raw_task_noises=torch.tensor([-9.1442, -6.5858, -6.5717]))
model.likelihood.initialize(raw_noise=torch.tensor([-7.4004]))
model.mean_module.initialize(constant=torch.tensor([9.2885]))
model.base_kernel.initialize(raw_lengthscale=torch.tensor([[-1.4706, -1.0452]]))
model.covar_module.initialize(raw_outputscale=torch.tensor([7.7791]))
model.covar_module.base_kernel.lengthscale = 0.209, 0.302

#Training the Model

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter=20

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.squeeze()[0],
        model.covar_module.base_kernel.lengthscale.squeeze()[1],
        model.likelihood.noise.item()
    ))
    optimizer.step()

#Making the hessian function (this was before class)

def mean_f(X):
        return likelihood(model(X)).mean
    
def P(Z,dim_num):
    X = Z.reshape(1,dim_num)
    tX = torch.autograd.Variable(X, requires_grad=True)
    dtX = torch.autograd.functional.jacobian(mean_f, tX)
    dtX = np.delete(dtX,0,1)
    dtX = dtX.reshape(dim_num,dim_num)
    return np.matrix(dtX)

model.eval()
likelihood.eval()

#Plot for determining c1 value

Z = torch.tensor([0.7, 0.7])
pk = linalg.solve(P(Z,2),np.negative(J(Z)))
c1 = 0.6
alpha = torch.linspace(0,1,100)
alpha = alpha.tolist()
instep = []
outstep = []
for i in range(100):
    instep.append(F(Z+(alpha[i])*pk))
    outstep.append(F(Z) + c1*alpha[i]*np.dot(J(Z),pk))
    
plt.plot(instep, 'r',outstep, 'b')

#Quasi-Newton Scheme

X = torch.tensor([0.7,0.7])
N = 100
dim_num = 2
J_min = 1e-6
c1 = 0.3
rho = 0.9
for i in range(N):
    func = F(X)
    grad = J(X)
    if linalg.norm(grad)<J_min:
        break
    pk = linalg.solve(P(X,2),np.negative(grad))
    a0 = 1
    while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk):
        a0 = a0*rho
    X = X + a0*pk
    print(i+1, '/', N)
    print(X,F(X),linalg.norm(J(X)),a0)
    
#If you have class GP installed you can use:
#(this has the model learn as it goes)

Y = torch.tensor([0.6,0.6])
GP.minimiseActive(F,J,Z,train_x,train_y,2,0.3,0.9)

#These minimsations give a lot of warnings but they still work fine and they go away for the 3D version of this code
