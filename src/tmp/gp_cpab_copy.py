import torch, gpytorch
import pdb, math, os, gc
from tqdm import tqdm

#from .gp_interpolation_old import gp_interpolation
from .gp_interpolation import gp_interpolation

from .template_cpab import template_cpab
#from .libcpab.libcpab import *
from .configManager import configManager

#from . import utilities


#from functools import singledispatch

#class gp_cpab(Cpab, gp_interpolation):
class gp_cpab(template_cpab, gp_interpolation):


    def __init__(self, tess_size, config, backend = 'numpy',
                        device = 'cpu', zero_boundary = True, 
                        volume_perservation = False, override = False):

        self.config = config
        self.constrain = self.config.parserinfo('*/noise_constraint')
        self.tasks = self.config.parserinfo('*/Tasks')
        self.interpolation_type = self.config.parserinfo('*/Interpolation_type')
        self.option = self.config.parserinfo('*/Option')
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = self.tasks, 
                                                                        noise_constraint = gpytorch.constraints.Interval(self.constrain[0],self.constrain[1]))

        pdb.set_trace()
        
        template_cpab.__init__(self,tess_size, self.config, backend=backend, device=device, zero_boundary=zero_boundary, 
                                    volume_perservation=volume_perservation, override=override)

        gp_interpolation.__init__(self,self.likelihood, self.config)



    def get_interpolation_inductive_points(self,X, multiout_Y, **kargs):
        pdb.set_trace()
        self.X_GP = X
        if 'padding_option' in kargs:
            setup_option = kargs['padding_option']#'padding_weights'
        else:
            setup_option = 'None'
            self.setup = 'None'


        if 'outsize' not in kargs:
            self.outsize_orig_data=(X.shape[1],X.shape[2])
        else:
            self.outsize_orig_data = kargs['outsize']
        self.multiout_Y_GP = multiout_Y
        if 'padded_idx' in kargs:
            self.padded_idx = kargs['padded_idx']
            self.non_padded_idx = list( set(range(0, self.outsize_orig_data[0])) - set(self.padded_idx) )
            partition_size = 1/(self.tasks-1)
            grid_split_size =  1/self.outsize_orig_data[0]
            lb_steps = 3
            self.padded_Y_ub = torch.tensor([partition_size]*(self.tasks)*len(self.padded_idx)).reshape(-1,len(self.padded_idx),self.tasks)
            self.padded_Y_lb = torch.tensor([partition_size]*(self.tasks)*(lb_steps-1)).reshape(-1,lb_steps-1,self.tasks) # 3 is the amount of padding info in the lower bound, the idea is to add info to anticipate to gaps at corners


            self.padding_grid_lb = ( torch.linspace(0,1,lb_steps) -1 )[:-1].reshape(1,1,lb_steps-1)
            #self.padding_grid_lb = torch.tensor([-3]*lb_steps)[:-1].reshape(1,1,lb_steps-1)
            self.padding_grid_ub = ( (torch.tensor(self.padded_idx) * (grid_split_size)) + 1).reshape(1,1,len(self.padded_idx))
            #self.non_padded_idx = [i + len(self.padding_grid_lb.flatten()) for i in self.non_padded_idx]
            #self.padded_idx = [i + len(self.padding_grid_lb.flatten()) for i in self.padded_idx]

            #self.tmp_grid_out_bound =  torch.tensor([(i* (1/ (self.outsize_orig_data[0] -1) ))+1 for i in range(1,len(self.padded_idx)+1)]).reshape(1,1,len(self.padded_idx))
            #self.tmp_grid_out_bound =  torch.tensor([(i* (1/ len(self.padded_idx) ))+1 for i in range(1,len(self.padded_idx)+1)]).reshape(1,1,len(self.padded_idx))
            #self.tmp_grid_out_bound =  torch.tensor([3]*len(self.padded_idx)).reshape(1,1,len(self.padded_idx))
            '''
            self.set_setup_input(padded_idx = self.padded_idx, nonpadded_idx = self.non_padded_idx, 
                                 padded_Y_lb = self.padded_Y_lb, padded_Y_ub = self.padded_Y_ub, 
                                 setup_scaling = setup_option, 
                                 padding_weights_lb = self.padding_grid_lb, 
                                 padding_weights_ub = self.padding_grid_ub)
            '''
            print('done')


    def determine_trans_dim(self, raw_data,modeflag):
        '''
        In pytorch:
            In 1D: [n_batch, n_channels, number_of_features]
            In 2D: [n_batch, n_channels, width, height]
        '''
        forw=[];rev=[];outsize=()
        if modeflag=='1D':
            forw=[0,2,1]; rev=[0,2,1]
            outsize=(raw_data.shape[1],raw_data.shape[2])
        else:
            forw=[0,3,1,2]; rev=[0,2,3,1] #=> height, width
            #forw=[0,3,2,1]; rev=[0,2,3,1]
            outsize=(raw_data.shape[1],raw_data.shape[2])
        return forw,rev,outsize

    def determine_trans_dim2(self, raw_data, ref, modeflag):
        '''
        In pytorch:
            In 1D: [n_batch, n_channels, number_of_features]
            In 2D: [n_batch, n_channels, width, height]
        '''
        forw,rev,outsize = self.determine_trans_dim(raw_data,modeflag)
        outsize=(ref.shape[1],ref.shape[2])
        return forw,rev,outsize




    def spatial_transformation(self, raw_data, ref, theta, modeflag):
        # Transform the images
        #pdb.set_trace()
        #forw,rev,outsize = utilities.determine_trans_dim2(raw_data,ref,modeflag)
        forw,rev,outsize = self.determine_trans_dim2(raw_data,ref,modeflag)

        if self.interpolation_type == 'linear':
            trans_data = self.transform_data(raw_data.permute(forw), theta, outsize=outsize)
            trans_data = trans_data.permute(rev)
            return trans_data, forw
        else:
            '''
            raw_data2 = torch.cat((raw_data,self.padded_Y),1)
            trans_data, sampled_data = self.transform_data(raw_data2.permute(forw), theta, outsize=outsize )
            '''
            trans_data, sampled_data = self.transform_data(raw_data.permute(forw), theta, outsize=outsize )
            trans_data = trans_data.permute(rev); sampled_data = sampled_data.permute(rev)
            return trans_data, sampled_data, forw


    ''' TO MAKE THE CHANGE BY USING DECORATORS BY @PROPERTY, @SETTER, @GETTER, THE CODE IS VERY MESSY AGAIN'''

    def make_grids_for_Regresion(self,**kargs):
        #pdb.set_trace()
        batch_size = kargs['batch_size']
        if self.setup == 'padding_weights':
            grid_no_expansion = self.uniform_meshgrid( (len(self.non_padded_idx),len(self.non_padded_idx))   ).repeat(batch_size,1,1)
            #grid_no_expansion = torch.cat( (self.padding_grid_lb.flatten(),grid_no_expansion.flatten(), self.padding_grid_ub.flatten()), 0)
            grid_no_expansion = torch.cat( (grid_no_expansion.flatten(), self.padding_grid_ub.flatten()), 0)
            grid_no_expansion = grid_no_expansion.reshape(batch_size,1,len(grid_no_expansion) )
        else:
            grid_no_expansion = self.uniform_meshgrid(self.outsize_orig_data).repeat(batch_size,1,1)

        return grid_no_expansion
    
    def transform_data(self, data, theta, outsize, **kargs):

        #self._check_type(data); self._check_device(data)
        #self._check_type(theta); self._check_device(theta)

        ''' Creation of Mesh Grid with same size as the reference output size'''

        # First, determine the batch size from original data, 
        # assuming [#batch, #channel, #features] per sequence 
        # and same batch size for theta, i.e. [#batch, theta_per_sequence]
        batch_size = data.shape[0]
        #pdb.set_trace()

        if self.interpolation_type == 'linear':
            #grid_t = self.transform_grid(self.grid, theta)
            #data_t = self.interpolate(data, grid_t, outsize)
            data_t = super(template_cpab, self).transform_data(data, theta, outsize)
            #data_t = super(Cpab, self).transform_data(data, theta, outsize)
            return data_t
        else:
            self._check_type(data); self._check_device(data)
            self._check_type(theta); self._check_device(theta)
            #pdb.set_trace()
            '''
            -----------------------------------------------------------------------------------------
            -----------------------------------------------------------------------------------------
            TEMPORARY BLOCK TO DEFINE THE NEW GRID RULES FOR SOLVING THE ISSUE REGARDING THE PADDING

            -----------------------------------------------------------------------------------------
            -----------------------------------------------------------------------------------------
            '''

            self.grid = self.uniform_meshgrid(outsize).repeat(batch_size,1,1)

            grid_no_expansion = self.make_grids_for_Regresion(batch_size = batch_size)
            
            '''
            if self.setup == 'padding_weights':
                grid_no_expansion = self.uniform_meshgrid( (len(self.non_padded_idx),len(self.non_padded_idx))   ).repeat(batch_size,1,1)
                grid_no_expansion = torch.cat( (grid_no_expansion.flatten(), self.tmp_grid_out_bound.flatten()), 0)
                grid_no_expansion = grid_no_expansion.reshape(batch_size,1,len(grid_no_expansion))
            else:
                grid_no_expansion = self.uniform_meshgrid(self.outsize_orig_data).repeat(batch_size,1,1)
            '''
            #grid_no_expansion = torch.cat( (grid_no_expansion,self.tmp_grid_out_bound) ,dim = 2)
            

            '''
            self.grid = self.uniform_meshgrid(outsize,batch_size = batch_size)
            grid_no_expansion = self.uniform_meshgrid(self.outsize_orig_data,batch_size = batch_size)
            '''
            
            '''
            -----------------------------------------------------------------------------------------
            -----------------------------------------------------------------------------------------
            -----------------------------------------------------------------------------------------
            -----------------------------------------------------------------------------------------
            '''
            # Necessary to add a validation in case there not exist a padding, pls, DONT FORGET!!!

            grid_t_no_expansion = self.transform_grid(grid_no_expansion, theta)
            ''' JUST HACKING THE EXAMPLE WITH 3 GAPS ---QLR FROM QLR--- AS BEGINNING'''
            #grid_t_no_expansion[:,:,2]=0.2; grid_t_no_expansion[:,:,3]=0.6
            '''
            -----------------------------------------------------------------------------------------
            -----------------------------------------------------------------------------------------
            -----------------------------------------------------------------------------------------
            -----------------------------------------------------------------------------------------
            '''

            '''grid_t_no_expansion = torch.cat( (grid_t_no_expansion,self.tmp_grid_out_bound) ,dim = 2)'''
            #data_t, sampled_data = self.interpolate_GP_1D_multitask(data, grid_t_no_expansion, outsize)
            #data_t, sampled_data = self.batch_interpolate_GP_1D_multitask(data, grid_t_no_expansion, outsize)
            data_t, sampled_data = self.batch_interpolate_GP_1D_multitask2(data, grid_t_no_expansion, outsize)

            #data_t, sampled_data = self.batch_interpolate_GP_1D_multitask2(data, grid_t_no_expansion, outsize, padded_idx = self.padded_idx, padded_Y = self.padded_Y)

            return data_t, sampled_data

