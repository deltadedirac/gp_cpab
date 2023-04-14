import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import pdb,gc

# Make plots inline

class GPbase(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPbase, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())  #gpytorch.kernels.MaternKernel(nu=1.5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)        #pdb.set_trace()
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):


    def __init__(self, train_x, train_y, likelihood, config, gp_init):
        super().__init__(train_x, train_y, likelihood)
        if len(gp_init)!=0:
            tasks, lengthscale, initialization = gp_init #gp_init['gp_params']
        else:
            #tasks, lengthscale, initialization = config.get_config_vals(['Tasks','Lengthscale','Initialization'])
            tasks, lengthscale, initialization = config.get_config_vals(['*/Tasks','*/Lengthscale','*/Initialization'])

        self.device = torch.device( 'cpu' ) #torch.device( config.get_config_vals(['device'])[0] )
        #tasks = config.parserinfo('*/Tasks')
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([tasks]), device=self.device)#.to(self.device)


        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(  batch_shape=torch.Size([tasks]) ),#lengthscale_prior=lengthscale_prior, 
            #outputscale_prior=outputscale_prior,
            batch_shape=torch.Size([tasks]),
            device=self.device
        )#.to(self.device)
        #import pdb; pdb.set_trace()
        self.covar_module.base_kernel.lengthscale =  lengthscale #config.parserinfo('*/Lengthscale')
        if isinstance(initialization,list):
            self.mean_module.initialize(constant= torch.tensor(initialization).reshape(-1,1).to(self.device) )
        else:
            self.mean_module.initialize(constant= initialization) #config.parserinfo('*/Initialization'))
        #self.mean_module.initialize(constant= initialization) #config.parserinfo('*/Initialization'))

    '''
    @property
    def init_prior(self):
        return self.initialization
    
    @init_prior.setter
    '''
    
    def forward(self, x):
        mean_x = self.mean_module(x)#.to(self.device)
        #pdb.set_trace()
        covar_x = self.covar_module(x)#.to(self.device)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 
        )




