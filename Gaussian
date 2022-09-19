#Preamble
import torch
import gpytorch
import math
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as linalg

%matplotlib inline
%load_ext autoreload
%autoreload 2

#Creating Data and Model

def Gauss(X,Y):
    a,b,c = 1,15,10 
    d,e,f = 1,15,10 
    u,v,w = 1,-15,10 
    x,y,z = 1,-15,10 
    alpha = torch.exp(-a*(X-b)**2/(2*c**2)-d*(Y-e)**2/(2*f**2))
    beta = torch.exp(-u*(X-v)**2/(2*w**2)-x*(Y-y)**2/(2*z**2))
    func = -alpha - beta
    dx = (a/c**2)*(X-b)*alpha + (u/w**2)*(X-v)*beta
    dy = (d/f**2)*(Y-e)*alpha + (x/z**2)*(Y-y)*beta
    return func, dx, dy

xv, yv = torch.meshgrid([torch.linspace(-10, 10, 10), torch.linspace(-10, 10, 10)])
train_x = torch.cat((
    xv.contiguous().view(xv.numel(), 1),
    yv.contiguous().view(yv.numel(), 1)),
    dim=1
)

f, dfx, dfy = Gauss(train_x[:, 0], train_x[:, 1])
train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)
train_y += 0.05 * torch.randn(train_y.size())

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

#Training the model

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05) 

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter=400

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

model.eval()
likelihood.eval()

#Creating Hessian from GP model

def mean_f(X):
    return likelihood(model(X)).mean

def P(Z):
    X = torch.stack([torch.tensor([Z[0]]), torch.tensor([Z[1]])], 1)
    tX = torch.autograd.Variable(X, requires_grad=True)
    dtX = torch.autograd.functional.jacobian(mean_f, tX)
    dtX = np.delete(dtX, 0)
    dtX = np.delete(dtX, 0)
    dtX = dtX.reshape(2,2)
    return np.matrix(dtX)
    
#Analytical function,jacobian, and hessian

def F(X):
    a,b,c = 1,15,10
    d,e,f = 1,15,10
    u,v,w = 1,-15,10
    x,y,z = 1,-15,10
    return -math.exp(-a*(X[0]-b)**2/(2*c**2)-d*(X[1]-e)**2/(2*f**2)) - math.exp(-u*(X[0]-v)**2/(2*w**2)-x*(X[1]-y)**2/(2*z**2))

def J(X):
    a,b,c = 1,15,10
    d,e,f = 1,15,10
    u,v,w = 1,-15,10
    x,y,z = 1,-15,10
    alpha = math.exp(-a*(X[0]-b)**2/(2*c**2)-d*(X[1]-e)**2/(2*f**2))
    beta = math.exp(-u*(X[0]-v)**2/(2*w**2)-x*(X[1]-y)**2/(2*z**2))
    dx = (a/c**2)*(X[0]-b)*alpha + (u/w**2)*(X[0]-v)*beta
    dy = (d/f**2)*(X[1]-e)*alpha + (x/z**2)*(X[1]-y)*beta
    return dx, dy
    
def H(X):
    a,b,c = 1,15,10
    d,e,f = 1,15,10
    u,v,w = 1,-15,10
    x,y,z = 1,-15,10
    alpha = math.exp(-a*(X[0]-b)**2/(2*c**2)-d*(X[1]-e)**2/(2*f**2))
    beta = math.exp(-u*(X[0]-v)**2/(2*w**2)-x*(X[1]-y)**2/(2*z**2))
    dxx = -(a**2/c**4)*(X[0]-b)**2*alpha + (a/c**2)*alpha - (u**2/w**4)*(X[0]-v)**2*beta + (u/w**2)*beta
    dxy = -(a/c**2)*(X[0]-b)*(d/f**2)*(X[1]-e)*alpha - (u/w**2)*(X[0]-v)*(x/z**2)*(X[1]-y)*beta
    dyx = -(a/c**2)*(X[0]-b)*(d/f**2)*(X[1]-e)*alpha - (u/w**2)*(X[0]-v)*(x/z**2)*(X[1]-y)*beta
    dyy = -(d**2/f**4)*(X[1]-e)**2*alpha + (d/f**2)*alpha - (x**2/z**4)*(X[1]-y)**2*beta + (x/z**2)*beta
    return np.matrix([[dxx,dxy],[dyx,dyy]])
    
#Newton Method, with path saved 

a0 = 1.
c1 = 0.15
c2 = 0.8
rho = 0.9
X = np.array([9.,9.])
N = 100
J_min = 1e-8

Xpath = [X]
Janal = [linalg.norm(J(X))]

for i in range(N):
    func = F(X)
    grad = J(X)
    if linalg.norm(grad)<J_min:
        break
    pk = linalg.solve(H(X),np.negative(grad))
    a0 = 1
    while not F(X+a0*pk) <= func + c1*a0*np.dot(grad,pk): #and np.dot(J(Y+a0*pk,px1,py1,px2,py2),pk) <= c2*np.dot(J(Y,px1,py1,px2,py2),pk):
            a0 = a0*rho
    X = X + a0*pk
    Xpath.append(X)
    Janal.append(linalg.norm(J(X)))
    print(i+1, '/', N)
    print(X,F(X),linalg.norm(J(X)),a0)
    
#Quasi-Newton Method with GP Hessian

a0 = 1.
c1 = 0.3
c2 = 0.8
rho = 0.8
Y = [9.,9.]
N = 100
J_min = 1e-8

Ypath = [Y]
Jmod = [linalg.norm(J(Y))]

for i in range(N):
    func = F(Y)
    grad = J(Y)
    if linalg.norm(grad)<J_min:
        break
    pk = linalg.solve(P(Y),np.negative(grad))
    a0 = 1
    while not F(Y+a0*pk) <= func + c1*a0*np.dot(grad,pk):
            a0 = a0*rho
    Y = Y + a0*pk
    Y = Y.tolist()
    Ypath.append(Y)
    Jmod.append(linalg.norm(J(Y)))
    print(i+1, '/', N)
    print(Y,F(Y),linalg.norm(J(Y)),a0)
    
#Plot of Paths 

def Gauss(X,Y,paramsx,paramsy):
    a,b,c = paramsx
    d,e,f = paramsy
    return -torch.exp(-a*(X-b)**2/(2*c**2)-d*(Y-e)**2/(2*f**2))

n1, n2 = 1000, 1000
x1,x2 = 5,25
xv, yv = torch.meshgrid([torch.linspace(x1, x2, n1), torch.linspace(x1, x2, n2)])
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
extent = (xv.min(), xv.max(), yv.max(), yv.min())
px1 = 1,15,10 
py1 = 1,15,10
px2 = 1,-15,10 
py2 = 1,-15,10
Gdata = Gauss(xv,yv,px1,py1) + Gauss(xv,yv,px2,py2)

ax.imshow(Gdata, extent=extent, cmap=cm.jet)
Xpath = np.array(Xpath)
Ypath = np.array(Ypath)
ax.plot(Xpath[:,0],Xpath[:,1], 'violet', marker = 'o', ms =5)
ax.plot(Ypath[:,0],Ypath[:,1], 'black', marker = 'x', ms = 5)
ax.set_title('Gauss')

#Gradient Magnitude vs Iterations

plt.semilogy(Janal, 'b', label = 'Analytic')
plt.semilogy(Jmod, 'r', label = 'model')
plt.legend()
