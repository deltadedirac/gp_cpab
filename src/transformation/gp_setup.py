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
        
        self.device = self.cast_device( config.parserinfo('device'))   #torch.device( 'cpu' ) 
        likelihood = likelihood.to(self.device)

        super().__init__(train_x, train_y, likelihood)
        if len(gp_init)!=0:
            tasks, lengthscale, initialization = gp_init #gp_init['gp_params']
        else:
            #tasks, lengthscale, initialization = config.get_config_vals(['Tasks','Lengthscale','Initialization'])
            tasks, lengthscale, initialization = config.get_config_vals(['*/Tasks','*/Lengthscale','*/Initialization'])

        #self.device = self.cast_device( config.parserinfo('device'))   #torch.device( 'cpu' ) 
        #tasks = config.parserinfo('*/Tasks')
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([tasks]), device = torch.device(self.device))


        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(  batch_shape=torch.Size([tasks]) ),#lengthscale_prior=lengthscale_prior, 
            #outputscale_prior=outputscale_prior,
            batch_shape=torch.Size([tasks]),
            device=torch.device(self.device)
        )#.to(self.device)
        #import pdb; pdb.set_trace()
        self.covar_module.base_kernel.lengthscale =  lengthscale #config.parserinfo('*/Lengthscale')
        if isinstance(initialization,list):
            self.mean_module.initialize(constant= torch.tensor(initialization, device = torch.device(self.device)).reshape(-1,1) )
        else:
            self.mean_module.initialize(constant= initialization, device = torch.device(self.device)) #config.parserinfo('*/Initialization'))
        #self.mean_module.initialize(constant= initialization) #config.parserinfo('*/Initialization'))

    '''
    @property
    def init_prior(self):
        return self.initialization
    
    @init_prior.setter
    '''
    def cast_device(self,device_tag):
        ''' convert the reference tag used on cpab for the ones enable in 
            pytorch for using different devices like cpu, cuda, mps, etc'''
        if type(device_tag)==str:
                casted_device = "cuda" if device_tag=="gpu" or device_tag=="cuda" else "cpu" if device_tag=='cpu' \
                                                                                    else 'mps'
        return casted_device
    
    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        mean_x = self.mean_module(x)#.to(self.device)
        #pdb.set_trace()
        covar_x = self.covar_module(x)#.to(self.device)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x.to(self.device), covar_x.to(self.device)) 
        )




