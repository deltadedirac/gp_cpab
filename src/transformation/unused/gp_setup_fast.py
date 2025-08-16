import math
import torch
import gpytorch
import pdb,gc

# Make plots inline

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):


    def __init__(self, train_x, train_y, likelihood, config, gp_init):
        
        # Setup the class initialization for GPs configuration
        self.device = self.cast_device( config.parserinfo('device'))
        likelihood = likelihood.to(self.device)

        if len(gp_init)!=0:
            tasks, lengthscale, initialization = gp_init
        else:
            tasks, lengthscale, initialization = config.get_config_vals(['*/Tasks',
                                                                         '*/Lengthscale',
                                                                         '*/Initialization'])

        # Step 1: Flatten batches and points
        train_x_flat = train_x.reshape(-1, 1)  # [4*4, 1] = [16, 1]
        train_y_flat = train_y.reshape(-1, tasks)   # [4*4, 5] = [16, 5]
        self.tasks = tasks
        super().__init__(train_x_flat, train_y_flat, likelihood)


        self.mean_module = gpytorch.means.ConstantMean(
                                        batch_shape = torch.Size([self.tasks]), 
                                        device = torch.device(self.device))


        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel( batch_shape = torch.Size([self.tasks]) ),
            batch_shape = torch.Size([self.tasks]),
            device=torch.device(self.device)
        )
        
        #import pdb; pdb.set_trace()
        self.covar_module.base_kernel.lengthscale =  lengthscale
        if isinstance(initialization,list):
            per_task_init = torch.tensor(initialization, device=self.device)
            self.mean_module.initialize( constant= per_task_init )
        else:
            self.mean_module.initialize(constant= initialization,
                                        device = torch.device(self.device)) 



    def cast_device(self,device_tag):
        ''' convert the reference tag used on cpab for the ones enable in 
            pytorch for using different devices like cpu, cuda, mps, etc'''
        if type(device_tag)==str:
                casted_device = "cuda" if device_tag=="gpu" \
                                        or device_tag=="cuda" \
                                        else "cpu" if device_tag=='cpu' \
                                        else 'mps'
        return casted_device
    
    
    def forward(self, x):

        #import ipdb; ipdb.set_trace()

        mean_x = self.mean_module(x)  # [N, T]
        covar_x = self.covar_module(x)  # [T, N, N]
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )





