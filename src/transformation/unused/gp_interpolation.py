import torch, gpytorch
import pdb, math, os, gc

#from utilities import determine_trans_dim,determine_trans_dim2
from .gp_setup import BatchIndependentMultitaskGPModel
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import LikelihoodList



class gp_interpolation:

    def __init__(self, likelihood_Multitask, config, **args_gp):
        self.config = config
        self.likelihood_Multitask = likelihood_Multitask
        self.padded_idx = None
        if 'gp_setup_params' in args_gp:
            self.args_gp_interpolation = args_gp['gp_setup_params']
        else:
            self.args_gp_interpolation = []

    
    def set_setup_input(self, **kargs):#padded_idx):
        self.padded_idx = kargs['padded_idx']
        self.non_padded_idx = kargs['nonpadded_idx']
        #self.Y_padded_lb = kargs['padded_Y_lb']
        #self.Y_padded_ub = kargs['padded_Y_ub']
        if 'setup_scaling' in kargs:
            self.setup = kargs['setup_scaling']
        else:
            self.setup = 'None'
        self.padding_weights_lb = kargs['padding_weights_lb']
        self.padding_weights_ub = kargs['padding_weights_ub']


    def sets_MultioutputGP_per_batches(self, x, data_sampled, batches):

        list_of_Multitask_GPs = []
        list_of_Multitask_Likelihood = []
        #self.constrain = self.config.parserinfo('*/noise_constraint')

        for i in range(0,batches):
            #likelihood_comp = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = self.tasks, 
            #                                                        noise_constraint = gpytorch.constraints.Interval(self.constrain[0],self.constrain[1]))
            list_of_Multitask_GPs.append( BatchIndependentMultitaskGPModel(x[i], data_sampled[i], self.likelihood_Multitask, self.config, self.args_gp_interpolation ) )
            list_of_Multitask_Likelihood.append(self.likelihood_Multitask)

        batch_model = IndependentModelList(*list_of_Multitask_GPs)
        batch_likelihood = LikelihoodList(*list_of_Multitask_Likelihood)

        batch_model.train()
        batch_likelihood.train()

        return batch_model, batch_likelihood




    def scale_indexes_grids2(self,grid, grid_to_transform, n_batch, width_to_trans, width):

        x = grid[:,0]

        # Do sampling
        x0 = torch.round(x * (width-1) ).to(torch.int64)
        # Clip values, USE LATER THE INDEX OPERATIONS IN TENSOR, JUST LOOK AT PYTORCH DOCS
        if self.setup == 'padding' or self.setup == 'padding_weights':
            #tmp_nonpadding = [ i + len(self.padding_weights_lb.flatten()) for i in self.non_padded_idx]
            #x0 = torch.clamp(x0[:,tmp_nonpadding], 0, (width-1))
            #x0 = torch.clamp(x0[:,self.non_padded_idx], 0, (width-1))
            #x0 =  [ torch.clamp(x0[i,self.non_padded_idx[i]], 0, (width-1)) for i in range(0,len(self.non_padded_idx)) ]
            ''' FIX OUT THIS, SPECIALLY IN NEGATIVE NUMBERS'''
            x0=x0


            #x0 = torch.stack(x0,1)
            #x0 = torch.clamp(x0, 0, (width-1))
        else:
            x0 = torch.clamp(x0, 0, (width-1))
            #x0=x0
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
            #CHANGE ALL THIS PART
            #data_sampled = data[batch_idx, :, x0[:,self.non_padded_idx]]
            #data_sampled = torch.cat((data_sampled, self.padded_Y_ub),1)

            #pdb.set_trace()

            '''
            tmp = []; tmp_join = []
            for i in range(0,batch_idx.shape[0]):
                tmp = data[batch_idx[i,0:len(self.non_padded_idx[i])], :, x0[i,self.non_padded_idx[i]]] 

                if len(self.padded_Y_ub[i]) > 0:
                    tmp_join.append( torch.cat((tmp, self.padded_Y_ub[i].reshape(1,-1))) )
                else:
                    tmp_join.append(tmp)
            '''

            tmp = []; tmp_join = []
            for i in range(0,batch_idx.shape[0]):
                #tmp = data[batch_idx[i,0:len(self.non_padded_idx[i])], :, x0[i,self.non_padded_idx[i]]]
                tmp = data[batch_idx[i,0:len(self.non_padded_idx[i])], :, self.non_padded_idx[i]] 
 

                if len(self.padding_weights_ub[i]) > 0:
                    tmp2 = data[batch_idx[i,0:len(self.padded_idx[i])], :, self.padded_idx[i]] 
                    tmp_join.append( torch.cat((tmp, tmp2)) )
                else:
                    tmp_join.append(tmp)


            data_sampled = torch.stack(tmp_join)

            #data_sampled = torch.cat((self.padded_Y_lb,data_sampled, self.padded_Y_ub),1)

        else: # NO PADDING OR JUST IGNORING THE PADDING, I.E. PADDING AS PART OF THE SEQUENCES. TEMPROARY SOLUTION WAS TO CLONNING WHAT IT IS ABOVE
            data_sampled = data[batch_idx, :, x0]

            '''TEMPORAL SOLUTION'''
            '''----------------------------------------------------------------------------------------------'''
            '''
            tmp = []; tmp_join = []
            for i in range(0,batch_idx.shape[0]):
                #tmp = data[batch_idx[i,0:len(self.non_padded_idx[i])], :, x0[i,self.non_padded_idx[i]]]
                tmp = data[batch_idx[i,0:len(self.non_padded_idx[i])], :, self.non_padded_idx[i]]

                if len(self.padding_weights_ub[i]) > 0:
                    tmp2 = data[batch_idx[i,0:len(self.padded_idx[i])], :, self.padded_idx[i]] 
                    tmp_join.append( torch.cat((tmp, tmp2)) )
                else:
                    tmp_join.append(tmp)
            data_sampled = torch.stack(tmp_join)
            '''
            '''----------------------------------------------------------------------------------------------'''

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
        multiout_GP_likelihood.eval()
        test_x =  [i.flatten() for i in test_points]
        #test_x =  [i for i in test_points]
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



