import torch, gpytorch
import pdb, math, os, gc

from utilities import *
from gp_setup import BatchIndependentMultitaskGPModel


class gp_interpolation:

    def __init__(self, likelihood_Multitask, config):
        self.config = config
        self.likelihood_Multitask = likelihood_Multitask

    
    def interpolate_GP_1D_multitask(self,data, grid_t_inv, outsize):
        # Problem size
        n_batch = data.shape[0]
        n_channels = data.shape[1]
        width = outsize[0]
        out_width = outsize[0]
        ''' 
            The CPAB inverse transform of the uniform grid 
            over the raw data would be the training points in GP
        '''
        x = grid_t_inv[:,0].flatten()
        # Do sampling
        x0 = torch.round(x * (data.shape[2]-1) ).to(torch.int64)
        # Clip values
        x0 = torch.clamp(x0, 0, (data.shape[2]-1))
                # Scale to domain
        x = x * (width-1)
        self.cpabgrid = x

        # Batch effect
        batch_size = len(x)
        batch_idx = torch.arange(n_batch).repeat(batch_size, 1).t().flatten()
        data_sampled = data[batch_idx, :, x0]
        #pdb.set_trace()
        self.multiout_GP_Interpolator = BatchIndependentMultitaskGPModel(x, data_sampled, self.likelihood_Multitask, self.config)

        # To get distribution over Posteriors
        trans_data = self.predict_operation(self.grid.flatten() * (width-1), self.multiout_GP_Interpolator)
        #trans_data_train = self.predict_operation(x, self.multiout_GP_Interpolator)

        with gpytorch.settings.fast_pred_var():#pdb.set_trace()
            cc = trans_data.mean.flatten()
            #ctrain = trans_data_train.mean.flatten()        
            '''JUST FOR SEEING WHAT IS GOING ON WITH MY TRAIN POINTS'''
            #self.ctrain = torch.reshape(ctrain, (n_batch, len(x), n_channels))

            self.lower,self.upper =  trans_data.confidence_region()
            #self.lower_train,self.upper_train =  trans_data_train.confidence_region()
            sampled_data = trans_data.rsample().unsqueeze(0).flatten()

        #gc.collect()
        #torch.cuda.empty_cache()
         # Reshape
        new_data = torch.reshape(cc, (n_batch, out_width, n_channels))
        sampled_data = torch.reshape(sampled_data, (n_batch, out_width, n_channels))
        new_data = new_data.permute(0, 2, 1)
        sampled_data = sampled_data.permute(0, 2, 1)

        return new_data.contiguous(), sampled_data


    def predict_operation(self, test_points, multiout_GP_Interpolator):
        multiout_GP_Interpolator.eval()
        self.likelihood_Multitask.eval()
        with gpytorch.settings.fast_pred_var():
            trans_data = self.likelihood_Multitask(multiout_GP_Interpolator(test_points))

        return trans_data