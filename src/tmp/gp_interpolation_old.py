import torch, gpytorch
import pdb, math, os, gc

#from utilities import determine_trans_dim,determine_trans_dim2
from ..transformation.gp_setup import BatchIndependentMultitaskGPModel
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import LikelihoodList



class gp_interpolation:

    def __init__(self, likelihood_Multitask, config):
        self.config = config
        self.likelihood_Multitask = likelihood_Multitask
        self.padded_idx = None
    
    def interpolate_GP_1D_multitask(self,data, grid_t_inv, outsize):
        # Problem size
        n_batch = data.shape[0]
        n_channels = data.shape[1]
        width = outsize[0]  # SUPERIMPORTANT THIS LINE, NEVER FORGET
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
        '''
        This specific chunk of code is just for case
        in which you have a reference and want to converted
        into a reference:
        '''
        batch_size = data.shape[2]
        
        #batch_size = out_width
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
            sampled_data = trans_data.rsample().flatten()

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


    def scale_indexes_grids(self,grid, grid_to_transform, n_batch, width_to_trans, width):

        ''' 
            The CPAB inverse transform of the uniform grid 
            over the raw data would be the training points in GP
        '''
        #pdb.set_trace()
        x = grid[:,0]
        # Do sampling
        x0 = torch.round(x * (width-1) ).to(torch.int64)
        # Clip values
        x0 = torch.clamp(x0, 0, (width-1)).flatten()
        x_trans = grid_to_transform
        for i in range(0,n_batch):
            x[i]+=2*i
            x_trans[i] +=2*i

        x = x.flatten()*(width_to_trans-1)
        x_trans = x_trans.flatten()*(width_to_trans-1)

        return x, x0, x_trans

        

    def batch_interpolate_GP_1D_multitask(self,data, grid_t_inv, outsize):
        # Problem size
        n_batch = data.shape[0]
        n_channels = data.shape[1]
        width = outsize[0]
        out_width = outsize[0]
        ''' 
            The CPAB inverse transform of the uniform grid 
            over the raw data would be the training points in GP
        '''
        #pdb.set_trace()
        x, x0, x_trans = self.scale_indexes_grids(grid_t_inv, self.grid, n_batch, width, data.shape[2])       

        batch_size = data.shape[2]
        batch_idx = torch.arange(n_batch).repeat(batch_size, 1).t().flatten()
        data_sampled = data[batch_idx, :, x0]
        #pdb.set_trace()
        self.multiout_GP_Interpolator = BatchIndependentMultitaskGPModel(x, data_sampled, self.likelihood_Multitask, self.config)

        # To get distribution over Posteriors
        trans_data = self.predict_operation(x_trans, self.multiout_GP_Interpolator)
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



    '''
    ---------------------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------
    '''


    def sets_MultioutputGP_per_batches(self, x, data_sampled, batches):

        list_of_Multitask_GPs = []
        list_of_Multitask_Likelihood = []
        self.constrain = self.config.parserinfo('*/noise_constraint')

        for i in range(0,batches):
            #likelihood_comp = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = self.tasks, 
            #                                                        noise_constraint = gpytorch.constraints.Interval(self.constrain[0],self.constrain[1]))
            list_of_Multitask_GPs.append( BatchIndependentMultitaskGPModel(x[i], data_sampled[i], self.likelihood_Multitask, self.config) )
            list_of_Multitask_Likelihood.append(self.likelihood_Multitask)

        batch_model = IndependentModelList(*list_of_Multitask_GPs)
        batch_likelihood = LikelihoodList(*list_of_Multitask_Likelihood)

        batch_model.train()
        batch_likelihood.train()

        return batch_model, batch_likelihood


    def set_setup_input(self, **kargs):#padded_idx):
        self.padded_idx = kargs['padded_idx']
        self.non_padded_idx = kargs['nonpadded_idx']
        self.Y_padded = kargs['padded_Y']
        if 'setup_scaling' in kargs:
            self.setup = kargs['setup_scaling']
        else:
            self.setup = 'None'
        self.padding_weights = kargs['padding_weights']


    def scale_indexes_grids2(self,grid, grid_to_transform, n_batch, width_to_trans, width):

        x = grid[:,0]

        # Do sampling
        x0 = torch.round(x * (width-1) ).to(torch.int64)
        # Clip values
        if self.setup == 'padding' or self.setup == 'padding_weights':
            x0 = torch.clamp(x0[:,self.non_padded_idx], 0, (width-1))
        else:
            x0 = torch.clamp(x0, 0, (width-1))
        '''
        elif self.setup == 'padding_weights':
            x0 = torch.clamp(x0[:,self.non_padded_idx], 0, (width-1))
            x = torch.cat((x[:, self.non_padded_idx], self.padding_weights.squeeze(0)),1)
        '''


        x_trans = grid_to_transform

        x = x *(width_to_trans-1)
        x_trans = x_trans *(width_to_trans-1)

        return x, x0, x_trans

    def build_Y_for_GPInterpolation(self,data, x0, batch_idx):
        if self.setup == 'padding' or self.setup == 'padding_weights':
            data_sampled = data[batch_idx, :, x0[:,self.non_padded_idx]]
            data_sampled = torch.cat((data_sampled, self.padded_Y),1)
        else:
            data_sampled = data[batch_idx, :, x0]
        
        return data_sampled

    def batch_interpolate_GP_1D_multitask2(self,data, grid_t_inv, outsize, **kargs):
        # Problem size
        n_batch = data.shape[0]
        n_channels = data.shape[1]
        width = outsize[0]
        out_width = outsize[0]


        #pdb.set_trace()
        x, x0, x_trans = self.scale_indexes_grids2(grid_t_inv, self.grid, n_batch, out_width, data.shape[2])       

        batch_size = data.shape[2]
        batch_idx = torch.arange(n_batch).repeat(batch_size, 1).t()

        #data_sampled = data[batch_idx, :, x0]
        data_sampled = self.build_Y_for_GPInterpolation(data, x0, batch_idx)


        #pdb.set_trace()

        batch_Multitask_model, batch_multitask_likelihood = self.sets_MultioutputGP_per_batches(x, data_sampled, n_batch)
        #self.multiout_GP_Interpolator = BatchIndependentMultitaskGPModel(x, data_sampled, self.likelihood_Multitask, self.config)

        # To get distribution over Posteriors
        trans_data, sampled_data, \
                self.lower,self.upper  = self.predict_operation2(x_trans, batch_Multitask_model, batch_multitask_likelihood  )
        #trans_data_train = self.predict_operation(x, self.multiout_GP_Interpolator)
        cc = trans_data.flatten()
        sampled_data = sampled_data.flatten()

        # Reshape
        new_data = torch.reshape(cc, (n_batch, out_width, n_channels))
        sampled_data = torch.reshape(sampled_data, (n_batch, out_width, n_channels))
        new_data = new_data.permute(0, 2, 1)
        sampled_data = sampled_data.permute(0, 2, 1)

        return new_data.contiguous(), sampled_data


    def predict_operation2(self, test_points, multiout_GP_Interpolator, multiout_GP_likelihood ):
        multiout_GP_Interpolator.eval()
        multiout_GP_likelihood .eval()
        test_x =  [i.flatten() for i in test_points]
        with gpytorch.settings.fast_pred_var():
            trans_data_distribution = multiout_GP_likelihood(*multiout_GP_Interpolator(*test_x))
            mean, posterior_samples, _lb, _ub = self.sampling_from_posterior(trans_data_distribution)

        return mean, posterior_samples, _lb, _ub

    def sampling_from_posterior(self, set_of_GP_distributions):

        list_GP_means = []      ; list_GP_Posterior_samples = []
        list_GP_lowerbound = [] ; list_GP_upperbound = []

        for i in set_of_GP_distributions:
            list_GP_means.append(i.mean)
            list_GP_Posterior_samples.append(i.rsample())
            l,u =i.confidence_region()
            list_GP_lowerbound.append(l)
            list_GP_upperbound.append(u)

        return torch.cat(list_GP_means), torch.cat(list_GP_Posterior_samples), \
                                    torch.cat(list_GP_lowerbound), torch.cat(list_GP_upperbound)



