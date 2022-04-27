import torch, gpytorch
import pdb, math, os, gc
from tqdm import tqdm

from gp_interpolation import gp_interpolation
from template_cpab import template_cpab

from utilities import *
from dataLoaderDiffeo import *
from configManager import *
from template_cpab import template_cpab
from gp_interpolation import gp_interpolation
#from functools import singledispatch


class gp_cpab(template_cpab, gp_interpolation):

    def __init__(self, tess_size, config, backend = 'numpy',
                        device = 'cpu', zero_boundary = True, 
                        volume_perservation = False, override = False):

        self.config = config
        constrain = self.config.parserinfo('*/noise_constraint')
        self.tasks = self.config.parserinfo('*/Tasks')
        self.interpolation_type = self.config.parserinfo('*/Interpolation_type')
        self.option = self.config.parserinfo('*/Option')
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = self.tasks, 
                                                                        noise_constraint = gpytorch.constraints.Interval(constrain[0],constrain[1]))

        template_cpab.__init__(self,tess_size, self.config, backend=backend, device=device, zero_boundary=zero_boundary, 
                                    volume_perservation=volume_perservation, override=override)

        gp_interpolation.__init__(self,likelihood, self.config)



    def get_interpolation_inductive_points(self,X, multiout_Y):
        #pdb.set_trace()
        self.X_GP = X
        self.outsize_orig_data=(X.shape[1],X.shape[2])
        self.multiout_Y_GP = multiout_Y

    def spatial_transformation(self, raw_data, ref, theta, modeflag):
        # Transform the images
        forw,rev,outsize = determine_trans_dim2(raw_data,ref,modeflag)
        if self.interpolation_type == 'linear':
            trans_data = self.transform_data(raw_data.permute(forw), theta, outsize=outsize)
            trans_data = trans_data.permute(rev)
            return trans_data, forw
        else:
            trans_data, sampled_data = self.transform_data(raw_data.permute(forw), theta, outsize=outsize)
            trans_data = trans_data.permute(rev); sampled_data = sampled_data.permute(rev)
            return trans_data, sampled_data, forw


    def transform_data(self, data, theta, outsize):

        self._check_type(data); self._check_device(data)
        self._check_type(theta); self._check_device(theta)

        ''' Creation of Mesh Grid with same size as the reference output size'''
        self.grid = self.uniform_meshgrid(outsize)


        if self.interpolation_type == 'linear':
            grid_t = self.transform_grid(self.grid, theta)
            data_t = self.interpolate(data, grid_t, outsize)
            return data_t
        else:
            grid_no_expansion = self.uniform_meshgrid(self.outsize_orig_data)
            grid_t_no_expansion = self.transform_grid(grid_no_expansion, theta)
            data_t, sampled_data = self.interpolate_GP_1D_multitask(data, grid_t_no_expansion, outsize)
            return data_t, sampled_data

