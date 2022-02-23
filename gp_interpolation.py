import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import pdb,gc

# Make plots inline

class GPInterpolation(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPInterpolation, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())  #gpytorch.kernels.MaternKernel(nu=1.5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)        #pdb.set_trace()
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):


    def __init__(self, train_x, train_y, likelihood, tasks):
        super().__init__(train_x, train_y, likelihood)
        '''self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([tasks]))'''
        self.mean_module = gpytorch.means.ConstantMean(prior=gpytorch.priors.NormalPrior(1/3,0.0001) ,batch_shape=torch.Size([tasks]))
        

        #lengthscale_prior = gpytorch.priors.NormalPrior(1./3., 1e-3)
        #outputscale_prior = gpytorch.priors.NormalPrior(1./3., 1e-3)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(  batch_shape=torch.Size([tasks]) ),#lengthscale_prior=lengthscale_prior, 
            #outputscale_prior=outputscale_prior,
            batch_shape=torch.Size([tasks])
        
        )

        #self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean
        #self.covar_module.outputscale = outputscale_prior.mean
        self.covar_module.base_kernel.lengthscale =  0.1
        #self.mean_module.initialize(constant= 1/3)

    def forward(self, x):
        mean_x = self.mean_module(x)
        #pdb.set_trace()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class BatchesGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, tasks):
        super(BatchesGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale = 10,batch_shape=torch.Size([tasks])),
            batch_shape=torch.Size([tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)





