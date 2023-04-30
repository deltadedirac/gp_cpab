import torch, gpytorch
import pdb, math, os, gc
from tqdm import tqdm
#from .gp_interpolation_old import gp_interpolation
from .gp_interpolation import gp_interpolation

from .template_cpab import template_cpab
#from .libcpab.libcpab import *
from .configManager import configManager



class gp_cpab(template_cpab, gp_interpolation):


    def __init__(self, tess_size, config, backend = 'numpy',
                        device = 'cpu', zero_boundary = True, 
                        volume_perservation = False, override = False, **arggp):
        params = tuple()
        self.config = config
        self.device = device

        if 'argparser_gpdata' in arggp:
            self.constrain, self.tasks, self.interpolation_type,  self.option , lengthscale, initialization  = arggp['argparser_gpdata']
            params = (self.tasks, lengthscale, initialization)
        else:
            self.constrain, self.tasks, self.interpolation_type, self.option = \
                                    self.config.get_config_vals(['*/noise_constraint', '*/Tasks', '*/Interpolation_type', '*/Option'])

        
        template_cpab.__init__(self,tess_size, self.config, backend=backend, device=device, zero_boundary=zero_boundary, 
                                    volume_perservation=volume_perservation, override=override)
        
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = self.tasks, 
                                                                           noise_constraint = gpytorch.constraints.Interval(self.constrain[0],self.constrain[1]) )\
                                                                                .to( self.cast_device(self.device) )
        

        gp_interpolation.__init__(self,self.likelihood, self.config, gp_setup_params=params)


    def spatial_transformation(self, raw_data, ref, theta, modeflag):

        forw,rev,outsize = self.determine_trans_dim(raw_data,ref,modeflag)

        trans_data, sampled_data = self.transform_data(raw_data.permute(forw), theta, outsize=outsize )
        trans_data = trans_data.permute(rev)
        if sampled_data is not None: sampled_data = sampled_data.permute(rev)
        return trans_data, sampled_data, forw


    ''' TO MAKE THE CHANGE BY USING DECORATORS BY @PROPERTY, @SETTER, @GETTER, THE CODE IS VERY MESSY AGAIN'''
    
    def transform_data(self, data, theta, outsize, **kargs):


        ''' Creation of Mesh Grid with same size as the reference output size'''

        # First, determine the batch size from original data, 
        # assuming [#batch, #channel, #features] per sequence 
        # and same batch size for theta, i.e. [#batch, theta_per_sequence]
        # This applied to Standard CPAB, because of interpolation object that it
        # is used inside of it.
        batch_size = data.shape[0]
        #pdb.set_trace()

        self._check_type(data); self._check_device(data)
        self._check_type(theta); self._check_device(theta)
        #pdb.set_trace()

        self.grid = self.uniform_meshgrid(outsize).repeat(batch_size,1,1)

        grid_no_expansion = self.grid

        self.grid = self.backend.to(self.grid, device=self.device)
        grid_no_expansion = self.backend.to(grid_no_expansion, device=self.device)
        grid_t_no_expansion = self.transform_grid(grid_no_expansion, theta)

        ''' This line of code had to be change because the previous "torch.solve in expm.py" 
            is completely deprecated and unsuported for version 2.0, besides this last one
            is not suported by macOS'''
        #grid_t_no_expansion = self.transform_grid(grid_no_expansion.to(torch.device('cpu')), 
        #                                          theta.to(torch.device('cpu'))).to(torch.device(self.device))

        if self.interpolation_type == 'linear':
            data_t = self.interpolate(data, grid_t_no_expansion, outsize)
            return (data_t, None)
        else:
            #data = data.to(torch.device(self.device))
            data = self.backend.to(data, device=self.device)
            data_t, sampled_data = self.batch_interpolate_GP_1D_multitask(data, grid_t_no_expansion, outsize)
            return data_t, sampled_data


