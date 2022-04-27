import torch, gpytorch
import pdb, math, os, gc

from utilities import *
from dataLoaderDiffeo import *
from configManager import *
from libcpab.libcpab import Cpab
#import gpytorch
from LossFunctionsAlternatives import LossFunctionsAlternatives
from gp_setup import BatchIndependentMultitaskGPModel
from functools import singledispatch


class template_cpab(Cpab):

    def __init__(self, tess_size, config, backend = 'numpy',
                        device = 'cpu', zero_boundary = True, 
                        volume_perservation = False, override = False):
        pdb.set_trace()
        super().__init__(tess_size, backend=backend, device=device, zero_boundary=zero_boundary, 
                                    volume_perservation=volume_perservation, override=override)

        self.config = config
        self.interpolation_type = config.parserinfo('*/Interpolation_type')



    def sample_transformation_with_prior(self, n_sample=1, mean=None, 
                                         length_scale=0.1, output_variance=1):
        """ Function for sampling smooth transformations. The smoothness is determined
            by the distance between cell centers. The closer the cells are to each other,
            the more the cell parameters should correlate -> smooth transistion in
            parameters. The covariance in the D-space is calculated using the
            squared exponential kernel.
                
        Arguments:
            n_sample: integer, number of transformation to sample
            mean: [d,] vector, mean of multivariate gaussian
            length_scale: float>0, determines how fast the covariance declines 
                between the cells 
            output_variance: float>0, determines the overall variance from the mean
        Output:
            samples: [n_sample, d] matrix. Each row is a independen sample from
                a multivariate gaussian
        """
        
        # Get cell centers and convert to backend type
        centers = self.backend.to(self.tesselation.get_cell_centers(), device=self.device)
        #pdb.set_trace()
        # Get distance between cell centers
        dist = self.backend.pdist(centers)
        
        # Make into a covariance matrix between parameters
        ppc = self.params.params_pr_cell
        cov_init = self.backend.zeros(self.params.D, self.params.D)
        
        for i in range(self.params.nC):
            for j in range(self.params.nC):
                # Make block matrix with large values
                block = 100*self.backend.maximum(dist)*self.backend.ones(ppc, ppc)
                # Fill in diagonal with actual values
                block[self.backend.arange(ppc), self.backend.arange(ppc)] = \
                    self.backend.repeat(dist[i,j], ppc)
                # Fill block into the large covariance
                cov_init[ppc*i:ppc*(i+1), ppc*j:ppc*(j+1)] = block
        
        # Squared exponential kernel
        cov_avees = output_variance**2 * self.backend.exp(-(cov_init / (2*length_scale**2)))

        # Transform covariance to theta space
        B = self.backend.to(self.params.basis, device=self.device)
        B_t = self.backend.transpose(B)
        cov_theta = self.backend.to(self.backend.matmul(B_t, self.backend.matmul(cov_avees, B)), device=self.device)
        
        U,S,V = torch.linalg.svd(cov_theta, full_matrices=False)

        # Sample
        samples = self.sample_transformation(n_sample, mean=mean, cov=torch.diag(S))
        return samples


