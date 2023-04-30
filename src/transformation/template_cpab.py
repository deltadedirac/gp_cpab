import torch, gpytorch
import pdb, math, os, gc
'''
from .utilities import determine_trans_dim,determine_trans_dim2
from ..dataLoaderDiffeo import *
from ..configManager import *
from ..libcpab.libcpab import Cpab
#import gpytorch
from ..LossFunctionsAlternatives import LossFunctionsAlternatives
from .gp_setup import BatchIndependentMultitaskGPModel
from functools import singledispatch
'''
from .libcpab.libcpab import Cpab
from .configManager import configManager


'''
from .gp_setup import BatchIndependentMultitaskGPModel
from .configManager import configManager
from .libcpab.libcpab import Cpab
'''
#from ..libcpab.libcpab import Cpab

class template_cpab(Cpab):

    def __init__(self, tess_size, config, backend = 'numpy',
                        device = 'cpu', zero_boundary = True, 
                        volume_perservation = False, override = False):

        super().__init__(tess_size, backend=backend, device=device, zero_boundary=zero_boundary, 
                                    volume_perservation=volume_perservation, override=override)

        self.config = config



    def _sample_transformation_with_prior(self, n_sample=1, mean=None, 
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
    
    def uniform_meshgrid(self, n_points):
        """ Constructs a meshgrid 
        Arguments:
            n_points: list, number of points in each dimension
        Output:
            grid: [ndim, nP] matrix of points, where nP = product(n_points)
        """
        return self._uniform_meshgrid(self.params.ndim,
                                             self.params.domain_min,
                                             self.params.domain_max,
                                             n_points, self.device)
    

    def _uniform_meshgrid(self, ndim, domain_min, domain_max, n_points, device='cpu'):
        #device = torch.device('cpu') if device=='cpu' else torch.device('cuda')
        device = torch.device('cpu') if device=='cpu' else torch.device('cuda') if device=='cuda' or device=='gpu' \
                                                                                    else torch.device('mps')
        lin = [torch.linspace(domain_min[i], domain_max[i], n_points[i], 
                            device=device) for i in range(ndim)]
        mesh = torch.meshgrid(lin[::-1])
        grid = torch.cat([g.reshape(1,-1) for g in mesh[::-1]], dim=0)
        return grid


    #%%
    def _check_device(self, x):
        """ Asssert that x is on the same device (cpu or gpu) as the class """
        assert self.backend.check_device(x, self.device), '''Input is placed on 
            device {0} but the class expects it to be on device {1}'''.format(
            str(x.device), self.device)

    def to(x, dtype=torch.float32, device=None):
        if type(device)==str:
            device = torch.device("cuda") if device=="gpu" or device=="cuda" else torch.device("cpu") if device=='cpu' \
                                                                                    else torch.device('mps')
        return torch.tensor(x, dtype=dtype, device=device)
    
    def cast_device(self,device_tag):
        ''' convert the reference tag used on cpab for the ones enable in 
            pytorch for using different devices like cpu, cuda, mps, etc'''
        if type(device_tag)==str:
                casted_device = "cuda" if device_tag=="gpu" or device_tag=="cuda" else "cpu" if device_tag=='cpu' \
                                                                                    else 'mps'
        return casted_device
    #%%
    def _check_input(self, tess_size, backend, device, 
                     zero_boundary, volume_perservation, override):
        """ Utility function used to check the input to the class.
            Not meant to be called by the user. """
        assert len(tess_size) > 0 and len(tess_size) <= 3, \
            '''Transformer only supports 1D, 2D or 3D'''
        assert type(tess_size) == list or type(tess_size) == tuple, \
            '''Argument tess_size must be a list or tuple'''
        assert all([type(e)==int for e in tess_size]), \
            '''All elements of tess_size must be integers'''
        assert all([e > 0 for e in tess_size]), \
            '''All elements of tess_size must be positive'''
        assert backend in ['numpy', 'tensorflow', 'pytorch'], \
            '''Unknown backend, choose between 'numpy', 'tensorflow' or 'pytorch' '''
        assert device in ['cpu', 'gpu', 'mps'], \
            '''Unknown device, choose between 'cpu' or 'gpu' '''
        if backend == 'numpy':
            assert device == 'cpu', '''Cannot use gpu with numpy backend '''
        assert type(zero_boundary) == bool, \
            '''Argument zero_boundary must be True or False'''
        assert type(volume_perservation) == bool, \
            '''Argument volume_perservation must be True or False'''
        assert type(override) == bool, \
            '''Argument override must be True or False '''


    def __repr__(self):
        output = '''
        CPAB transformer class. 
            Parameters:
                Tesselation size:           {0}
                Total number of cells:      {1}
                Theta size:                 {2}
                Domain lower bound:         {3}
                Domain upper bound:         {4}
                Zero Boundary:              {5}
                Volume perservation:        {6}
            Backend:                        {7}
        '''.format(self.params.nc, self.params.nC, self.params.d, 
            self.params.domain_min, self.params.domain_max, 
            self.params.zero_boundary, self.params.volume_perservation,
            self.backend_name)
        return output