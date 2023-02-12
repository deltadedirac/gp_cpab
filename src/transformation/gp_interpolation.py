import torch, gpytorch
import pdb, math, os, gc

from .gp_setup import BatchIndependentMultitaskGPModel
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import LikelihoodList


class gp_interpolation:

    def __init__(self, likelihood_Multitask, config, **args_gp):
        self.config = config
        self.likelihood_Multitask = likelihood_Multitask

        self.args_gp_interpolation = ( [] , args_gp['gp_setup_params'] )[ 'gp_setup_params' in args_gp ]



    def sets_MultioutputGP_per_batches(self, x, data_sampled, batches):

        list_of_Multitask_GPs = []
        list_of_Multitask_Likelihood = []
        #self.constrain = self.config.parserinfo('*/noise_constraint')

        for i in range(0,batches):
            list_of_Multitask_GPs.append( BatchIndependentMultitaskGPModel(x[i], data_sampled[i], \
                                                                            self.likelihood_Multitask, \
                                                                            self.config, self.args_gp_interpolation ) )
            list_of_Multitask_Likelihood.append(self.likelihood_Multitask)

        batch_model = IndependentModelList(*list_of_Multitask_GPs)
        batch_likelihood = LikelihoodList(*list_of_Multitask_Likelihood)

        batch_model.train()
        batch_likelihood.train()

        return batch_model, batch_likelihood


    def scale_indexes_grids(self,grid, grid_to_transform, n_batch, width_to_trans, width):

        x = grid[:,0]

        # Do sampling
        x0 = torch.round(x * (width-1) ).to(torch.int64)
        # Clip values, USE LATER THE INDEX OPERATIONS IN TENSOR, JUST LOOK AT PYTORCH DOCS
        x_trans = grid_to_transform

        x = x *(width_to_trans-1)
        x_trans = x_trans *(width_to_trans-1)

        return x, x0, x_trans


    def batch_interpolate_GP_1D_multitask(self,data, grid_t_inv, outsize, **kargs):
        # Problem size
        n_batch = data.shape[0]
        n_channels = data.shape[1]
        width = outsize[0]
        out_width = outsize[0]

        x, x0, x_trans = self.scale_indexes_grids(grid_t_inv, self.grid, n_batch, out_width, data.shape[2])       

        batch_size = data.shape[2]
        batch_idx = torch.arange(n_batch).repeat(batch_size, 1).t()
        
        # could be more elegant this line by quiting the permuting in gp_cpab module, and just to plug it into the linear case, because here it is not necesarry
        data_sampled = data.permute(0,2,1) 
        batch_Multitask_model, batch_multitask_likelihood = self.sets_MultioutputGP_per_batches(x, data_sampled, n_batch)

        # To get distribution over Posteriors
        trans_data, sampled_data, \
                self.lower,self.upper  = self.predict_operation(x_trans, batch_Multitask_model, batch_multitask_likelihood  )
        cc = trans_data.flatten()
        sampled_data = sampled_data.flatten()

        # Reshape
        new_data = torch.reshape(cc, (n_batch, out_width, n_channels))
        sampled_data = torch.reshape(sampled_data, (n_batch, out_width, n_channels))
        new_data = new_data.permute(0, 2, 1)
        sampled_data = sampled_data.permute(0, 2, 1)

        return new_data.contiguous(), sampled_data


    def predict_operation(self, test_points, multiout_GP_Interpolator, multiout_GP_likelihood ):
        multiout_GP_Interpolator.eval()
        multiout_GP_likelihood.eval()
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


