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
        if 'argparser_gpdata' in arggp:
            self.config = config
            self.constrain, self.tasks, self.interpolation_type,  self.option , lengthscale, initialization  = arggp['argparser_gpdata']
            params = (self.tasks, lengthscale, initialization)
        else:
            self.config = config
            self.constrain = self.config.parserinfo('*/noise_constraint')
            self.tasks = self.config.parserinfo('*/Tasks')
            self.interpolation_type = self.config.parserinfo('*/Interpolation_type')
            self.option = self.config.parserinfo('*/Option')
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = self.tasks, 
                                                                        noise_constraint = gpytorch.constraints.Interval(self.constrain[0],self.constrain[1]))
        
        template_cpab.__init__(self,tess_size, self.config, backend=backend, device=device, zero_boundary=zero_boundary, 
                                    volume_perservation=volume_perservation, override=override)

        gp_interpolation.__init__(self,self.likelihood, self.config, gp_setup_params=params)




    
    def get_interpolation_inductive_points(self,X, multiout_Y, **kargs):
        #pdb.set_trace()
        self.raw_data = X
        if 'padding_option' in kargs:
            setup_option = kargs['padding_option']#'padding_weights'
            self.setup = setup_option #  SEE THE REASON WHY WITH THIS ENABLED IN THIS PART IS NOT WORKING PROPERLY, THAT DOESN;T MAKE SENSE
        else:
            setup_option = 'None'
            self.setup = 'None'


        if 'outsize' not in kargs:
            self.outsize_orig_data=(X.shape[1],X.shape[2])
        else:
            self.outsize_orig_data = kargs['outsize']
        if 'separation_size_grid' in kargs:
            grid_split_size = kargs['separation_size_grid']
        else:
            grid_split_size = 1/self.outsize_orig_data[0]
        self.multiout_Y_GP = multiout_Y
        if 'padded_idx' in kargs and setup_option !='none':
            self.padded_idx = kargs['padded_idx']
            if 'non_padded_idx' in kargs:
                self.non_padded_idx = kargs['non_padded_idx']
            else:
                self.non_padded_idx = list( set(range(0, self.outsize_orig_data[0])) - set(self.padded_idx) )

            partition_size = 1/(self.tasks-1)
            #grid_split_size =  1/self.outsize_orig_data[0]
            lb_steps = 3
            
            #self.padded_Y_ub = [ torch.tensor([partition_size]*(self.tasks)*len(i)).reshape(-1,len(i),self.tasks) if bool(i) else [] for i in self.padded_idx ] 
            #self.padded_Y_lb = [ torch.tensor([partition_size]*(self.tasks)*(lb_steps-1)).reshape(-1,lb_steps-1,self.tasks) for i in range(0,len(self.padded_idx)) ]# 3 is the amount of padding info in the lower bound, the idea is to add info to anticipate to gaps at corners

            self.padding_grid_lb =  [ ( torch.linspace(0,1,lb_steps) -1 )[:-1].reshape(1,1,lb_steps-1) ]*len(self.padded_idx)
            #self.padding_grid_ub = [ ( ( torch.tensor(i) * grid_split_size ) + 1).reshape(1,1,len(i)) if bool(i) else torch.tensor([]) for i in self.padded_idx ]
            self.padding_grid_ub = [ ( ( torch.tensor(i[0]) * 1./(len(i[1])-1) ) ).reshape(1,1,len(i[0])) if bool(i[0]) else torch.tensor([]) 
                                                                                                                for i in list(zip(self.padded_idx, self.non_padded_idx)) ]

            
            self.set_setup_input(padded_idx = self.padded_idx, nonpadded_idx = self.non_padded_idx, 
                                 #padded_Y_lb = self.padded_Y_lb, padded_Y_ub = self.padded_Y_ub, 
                                 setup_scaling = setup_option, 
                                 padding_weights_lb = self.padding_grid_lb, 
                                 padding_weights_ub = self.padding_grid_ub)
            
            #print('done')
    
    
    def determine_trans_dim(self, *args):
        '''
        In pytorch:
            In 1D: [n_batch, n_channels, number_of_features]
            In 2D: [n_batch, n_channels, width, height]
        '''

        forw=[];rev=[];outsize=()
        assert len(args) >= 1, "number parameters must not be 0"
        
        if len(args)==1:
            raw_data = args
            outsize=(raw_data.shape[1],raw_data.shape[2])
        else:
            raw_data, ref, modeflags = args
            outsize=(ref.shape[1],ref.shape[2])

        if self.params.ndim == 1:
            forw=[0,2,1]; rev=[0,2,1]
        else:
            forw=[0,3,1,2]; rev=[0,2,3,1] #=> height, width

        return forw,rev,outsize


    def spatial_transformation(self, raw_data, ref, theta, modeflag):

        forw,rev,outsize = self.determine_trans_dim(raw_data,ref,modeflag)

        '''
        if self.interpolation_type == 'linear':
            trans_data = self.transform_data(raw_data.permute(forw), theta, outsize=outsize)
            trans_data = trans_data.permute(rev)
            return trans_data, forw
        else:
        '''
        trans_data, sampled_data = self.transform_data(raw_data.permute(forw), theta, outsize=outsize )
        trans_data = trans_data.permute(rev)
        if sampled_data is not None: sampled_data = sampled_data.permute(rev)
        return trans_data, sampled_data, forw


    ''' TO MAKE THE CHANGE BY USING DECORATORS BY @PROPERTY, @SETTER, @GETTER, THE CODE IS VERY MESSY AGAIN'''

    def make_grids_for_Regresion(self,**kargs):
        #pdb.set_trace()
        batch_size = kargs['batch_size']
        if self.setup == 'padding_weights':

            '''
            grid_no_expansion = self.uniform_meshgrid( (len(self.non_padded_idx),len(self.non_padded_idx))   ).repeat(batch_size,1,1)
            grid_no_expansion = torch.cat( (grid_no_expansion.flatten(), self.padding_grid_ub.flatten()), 0)
            grid_no_expansion = grid_no_expansion.reshape(batch_size,1,len(grid_no_expansion) )
            '''
            #pdb.set_trace()
            grid_no_expansion = [ self.uniform_meshgrid( (len(p_idx),len(p_idx))   )  for p_idx in self.non_padded_idx ]

            grid_no_expansion = [  torch.cat( [ grid_no_expansion[i].flatten() , self.padding_grid_ub[i].flatten() ] ,0 )  \
                                                                                    for i in range(0,len(grid_no_expansion) ) ]

            grid_no_expansion = torch.stack(grid_no_expansion,0).reshape(batch_size,1,len(grid_no_expansion[0]) )

        else:
            grid_no_expansion = self.uniform_meshgrid(self.outsize_orig_data).repeat(batch_size,1,1)

        return grid_no_expansion
    
    def transform_data(self, data, theta, outsize, **kargs):


        ''' Creation of Mesh Grid with same size as the reference output size'''

        # First, determine the batch size from original data, 
        # assuming [#batch, #channel, #features] per sequence 
        # and same batch size for theta, i.e. [#batch, theta_per_sequence]
        batch_size = data.shape[0]
        #pdb.set_trace()
        '''
        if self.interpolation_type == 'linear':# and self.setup != 'padding_weights':
            #grid_t = self.transform_grid(self.grid, theta)
            #data_t = self.interpolate(data, grid_t, outsize)
            data_t = super(template_cpab, self).transform_data(data, theta, outsize)
            #data_t = super(Cpab, self).transform_data(data, theta, outsize)
            return (data_t, None)
        else:
        '''
        self._check_type(data); self._check_device(data)
        self._check_type(theta); self._check_device(theta)
        #pdb.set_trace()

        self.grid = self.uniform_meshgrid(outsize).repeat(batch_size,1,1)

        
        #pdb.set_trace()

        grid_no_expansion = self.make_grids_for_Regresion(batch_size = batch_size)
        grid_t_no_expansion = self.transform_grid(grid_no_expansion, theta)
        ''' JUST HACKING THE EXAMPLE WITH 3 GAPS ---QLR FROM QLR--- AS BEGINNING'''
        #grid_t_no_expansion[:,:,2]=0.2; grid_t_no_expansion[:,:,3]=0.6
        '''
        -----------------------------------------------------------------------------------------
        -----------------------------------------------------------------------------------------
        -----------------------------------------------------------------------------------------
        -----------------------------------------------------------------------------------------
        '''

        if self.interpolation_type == 'linear':
            data_t = self.interpolate(data, grid_t_no_expansion, outsize)
            return (data_t, None)
        else:
            data_t, sampled_data = self.batch_interpolate_GP_1D_multitask2(data, grid_t_no_expansion, outsize)
            #self.grid =   grid_t_no_expansion
            #data_t, sampled_data = self.batch_interpolate_GP_1D_multitask2(data, grid_no_expansion, outsize)

            return data_t, sampled_data


