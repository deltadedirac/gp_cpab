import torch, gpytorch
import pdb, math, os, gc

from .gp_setup_fast_3 import BatchIndependentMultitaskGPModel
from .gp_interpolation import gp_interpolation


class gp_interpolation(gp_interpolation):

    def __ini__(self, 
                likelihood_Multitask, 
                config, 
                **args_gp):
        
        super().__init__(likelihood_Multitask, 
                         config, 
                         **args_gp)
        
    def sets_MultioutputGP_per_batches(self, x, data_sampled, batches):

        #import ipdb; ipdb.set_trace()

        # --- Shape Parameters ---
        # S -> sequences
        # L -> grid points per sequence
        # T -> tasks (output dimensions)
        S, L, T = data_sampled.shape

        # --- Likelihood ---
        self.likelihood_Multitask = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=torch.Size([S * T]),
                noise_prior = gpytorch.priors.NormalPrior(loc=0, scale=1e-6), 
                has_global_noise= False, 
                has_task_noise=True, 
                noise_constraint = gpytorch.constraints.Interval(1e-6,1e-1)
        ).to( 
            self.cast_device(self.config.parserinfo('device')) 
            )


        batch_model = BatchIndependentMultitaskGPModel(
                                                    x, 
                                                    data_sampled, 
                                                    self.likelihood_Multitask,
                                                    self.config, 
                                                    self.args_gp_interpolation
                                        ).to( self.cast_device( 
                                                self.config.parserinfo('device')
                                                ) )

        #import ipdb; ipdb.set_trace()
        batch_model.train()
        self.likelihood_Multitask.train()

        return batch_model, self.likelihood_Multitask
    

    def predict_operation(self, 
                          test_points, 
                          multiout_GP_Interpolator, 
                          multiout_GP_likelihood ):
        
        multiout_GP_Interpolator.eval()
        multiout_GP_likelihood.eval()

        #import ipdb; ipdb.set_trace()

        '''
            the inclusion of fast_pred_samples as well as fast_computation a.k.a Cholesky
            was necessary to improve the speed performance of MOGP interpolator in the
            transformation. Otherwise, we can switch to just fast_pred_var

            Since we are using a Multioutput-GP as a bank of independent
            GPs for interpolating the states i.e. aminoacid channels, the
            last option (deterministic_probes) seems suitable for getting
            more stability in the convergence output value. That being said,
            It will be given as default, but that could be changed in the
            future

            - Important Note: THe inclusion of skipping posterior variance into
            the calculations for doing the regression, eliminate the uncertainty
            bounds in order to speed up the diffeomorphic transformation when long
            sequences are being aligning. This would affect the way to calculate the
            posterior variance for 2.1_toy_example       
        '''
        # It is necessary in the future to include the possibility to
        # enable/disable the posterior variance for calculations.
        # Just include it into the constructor

        
        with gpytorch.settings.fast_pred_var(True),\
            gpytorch.settings.fast_pred_samples(True),\
            gpytorch.settings.fast_computations(covar_root_decomposition=True, 
                                            log_prob=True),\
            gpytorch.settings.skip_posterior_variances(self.skip_posterior_variance), \
            gpytorch.settings.max_cg_iterations(5000),\
            gpytorch.settings.deterministic_probes(True),\
            gpytorch.settings.trace_mode(True):
        
        #with gpytorch.settings.fast_pred_var():    
            #import ipdb; ipdb.set_trace()

            test_x = multiout_GP_Interpolator._reformat_features(test_points.squeeze(1))
            trans_data_distribution = multiout_GP_likelihood(
                                                        multiout_GP_Interpolator(test_x)
                                                        )
            test_shape = torch.Size([
                multiout_GP_Interpolator.S,
                multiout_GP_Interpolator.T
            ])

            mean, posterior_samples, _lb, _ub = \
                        self.sampling_from_posterior(trans_data_distribution, test_shape)
        
        return mean, posterior_samples, _lb, _ub


    def sampling_from_posterior(self, 
                                set_of_GP_distributions, test_shape):

        #import ipdb; ipdb.set_trace()

        GP_means = set_of_GP_distributions.mean.reshape(*test_shape,-1).permute(0,2,1)
        lower,upper = set_of_GP_distributions.confidence_region()
        lower = lower.reshape(*test_shape,-1).permute(0, 2, 1)
        upper = upper.reshape(*test_shape,-1).permute(0, 2, 1)

        return GP_means, \
                set_of_GP_distributions.rsample().reshape(*test_shape,-1) ,\
                lower, upper

