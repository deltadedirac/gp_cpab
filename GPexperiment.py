# GPyTorch Imports
import gpytorch
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, MultitaskKernel
from gpytorch.kernels import RBFKernel, RBFKernel, ProductKernel
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList, MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal

# PyTorch
import torch

# Math, avoiding memory leak, and timing
import math
import gc
import math

class BatchedGP(ExactGP):

    def __init__(self, train_x, train_y, likelihood, shape, output_device, use_ard=False):

        # Run constructor of superclass
        super(BatchedGP, self).__init__(train_x, train_y, likelihood)

        # Determine if using ARD
        ard_num_dims = None
        if use_ard:
            ard_num_dims = train_x.shape[-1]

        # Create the mean and covariance modules
        self.shape = torch.Size([shape])
        self.mean_module = ConstantMean(batch_shape=self.shape)
        self.base_kernel = RBFKernel(batch_shape=self.shape,
                                        ard_num_dims=ard_num_dims)
        self.covar_module = ScaleKernel(self.base_kernel,
                                        batch_shape=self.shape,
                                        output_device=output_device)

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.
        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.
        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        mean_x = self.mean_module(x)  # Compute the mean at x
        covar_x = self.covar_module(x)  # Compute the covariance at x
        return MultivariateNormal(mean_x, covar_x)


"""Tester script for GPyTorch using analytic sine functions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error as mse
import pdb

def preprocess_eval_inputs(Zs, d_y, device="cpu"):
    """Helper function to preprocess inputs for use with training
    targets and evaluation.
    Parameters:
        Zs (np.array): Array of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        d_y (int):  The dimensionality of the targets of GPR.
        device (str):  Device to output the tensor to.
    Returns:
        eval_x (torch.tensor):  Torch tensor of shape (B * YD, N, XD).  This
            tensor corresponding to a tiled set of inputs is used as input for
            the inference model in FP32 format.
    """
    # Preprocess batch data
    eval_x = torch.tensor(Zs, device=device).double()
    eval_x = eval_x.repeat((d_y, 1, 1))
    return eval_x


if __name__ == "__main__":
    """Main tester function."""
    # Set parameters
    B = 256  # Number of batches
    N = 100  # Number of data points in each batch
    D = 3  # Dimension of X and Y data
    Ds = 1  # Dimensions for first factored kernel - only needed if factored kernel is used
    EPOCHS = 1  # Number of iterations to perform optimization
    THR = -1e5  # Training threshold (minimum)
    USE_CUDA = torch.cuda.is_available()  # Hardware acceleraton
    MEAN = 0  # Mean of data generated
    SCALE = 1  # Variance of data generated
    COMPOSITE_KERNEL = False  # Use a factored kernel
    USE_ARD = True  # Use Automatic Relevance Determination in kernel
    LR = 0.5  # Learning rate

    pdb.set_trace()

    # Create training data and labels
    train_x_np = np.random.normal(loc=MEAN, scale=SCALE, size=(B, N, D))  # Randomly-generated data
    train_y_np = np.sin(train_x_np)  # Analytic sine function

    Zs = train_x_np
    Ys = train_y_np
    B, N, XD = Zs.shape
    YD = train_y_np.shape[-1]
    batch_shape = B * YD

    # Format the training features - tile and reshape
    train_x = torch.tensor(Zs, device='cpu')
    # Now evaluate
    test_x_np = np.random.normal(loc=MEAN, scale=SCALE, size=(B, 1, D))
    train_x = preprocess_eval_inputs(train_x_np, train_y_np.shape[-1])
    test_x = preprocess_eval_inputs(test_x_np, train_y_np.shape[-1])

    # Format the training labels - reshape
    train_y = torch.vstack(
        [torch.tensor(Ys, device='cpu')[..., i] for i in range(YD)])

    # initialize likelihood and model
    likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_shape]))

    print('done')

    model = BatchedGP(train_x, train_y, likelihood, batch_shape,
                          'cpu', use_ard=USE_ARD)

    model_hyperparams=None
    model.initialize(**model_hyperparams)


    f_preds = model(test_x)
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean
    variance = observed_pred.variance
