import math
import torch, gpytorch
import pdb,gc
from gpytorch.lazy import BlockDiagLazyTensor
# Make plots inline

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):


    def __init__(self, train_x, train_y, likelihood, config, gp_init):
        
        # Setup the class initialization for GPs configuration
        self.device = self.cast_device( config.parserinfo('device'))
        likelihood = likelihood.to(self.device)

        if len(gp_init)!=0:
            tasks, lengthscale, initialization = gp_init
        else:
            tasks, lengthscale, initialization = config.get_config_vals(['*/Tasks',
                                                                         '*/Lengthscale',
                                                                         '*/Initialization'])

        #import ipdb; ipdb.set_trace()
        
        # --- Shape Parameters ---
        # S -> sequences
        # L -> grid points per sequence
        # T -> tasks (output dimensions)
        # N = L -> number of points per GP
        # D -> shape of each input point of X. As we are dealing with one-hot
        #      sequences, we are using 1D grids to represent X and one-hot encoded
        #      Y, where
        self.S, self.L, self.T = train_y.shape
        self.N = self.L
        self.D = 1

        train_x_reshaped = self._reformat_features(train_x)
        train_y_reshaped = self._reformat_SoftLabels(train_y)

        super().__init__(train_x_reshaped, train_y_reshaped, likelihood)


        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape = torch.Size([self.S * self.T]),
            device = torch.device(self.device)
        )

        self.covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.RBFKernel(
                                        batch_shape = torch.Size([self.S * self.T]),
                                        ard_num_dims=self.D
                                            ),
                                    batch_shape=torch.Size([self.S * self.T]),
                                    device=torch.device(self.device)
                                    )

        task_means = torch.tensor(initialization)
        batch_means = task_means.repeat(self.S)

        self.covar_module.base_kernel.lengthscale =  lengthscale
        self.mean_module.initialize(constant=batch_means)




    def _reformat_features(self, X):
        # Repeat each sequence’s grid for all T tasks
        # grid[:, None, :, None] = [S, 1, L, 1]
        # repeat → [S, T, L, 1] → reshape to [S*T, L, 1]
        #import ipdb; ipdb.set_trace()
        X_reshaped = X[:, None, :, None].repeat(1, 
                                                self.T, 
                                                1, 1).reshape(self.S * self.T, -1, 1)#reshape(self.S * self.T, self.L, 1)
        return X_reshaped
    
    def _reformat_SoftLabels(self, Y_raw):
        # Reshape Y to [S*T, L]
        Y = Y_raw.permute(0, 2, 1).reshape(self.S * self.T, self.L)
        return Y

    def cast_device(self,device_tag):
        ''' convert the reference tag used on cpab for the ones enable in 
            pytorch for using different devices like cpu, cuda, mps, etc'''
        if type(device_tag)==str:
                casted_device = "cuda" if device_tag=="gpu" \
                                        or device_tag=="cuda" \
                                        else "cpu" if device_tag=='cpu' \
                                        else 'mps'
        return casted_device
    

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        mean_x = self.mean_module(x)                 # [S*T, L]
        covar_x = self.covar_module(x)               # [S*T, L, L]
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


if __name__ == "__main__":

    #import ipdb; ipdb.set_trace()
    # --- Parameters ---
    S = 4   # sequences
    L = 4   # grid points per sequence
    T = 5   # tasks (output dimensions)
    N = L   # number of points per GP

    # --- Input Grid [L]: Shared across sequences ---
    grid_positions = torch.arange(L).float().view(-1, 1)  # [4, 1]

    # --- Create batched input [S*T, L, 1] ---
    # Repeat grid for each (sequence, task)
    X = grid_positions.expand(S * T, -1, -1).contiguous()  # [S*T, L, 1]

    grid_t = torch.tensor(
        [[[-0.0110,  0.3326,  0.6699,  0.9961]], 
        [[ 0.0819,  0.6624,  1.2945,  1.9481]],
        [[ 0.0356,  0.3366,  1.1352,  1.8277]],  
        [[ 0.0151,  0.6391,  1.0329,  1.5070]]]).squeeze(1)
    
    # Repeat each sequence’s grid for all T tasks
    # grid[:, None, :, None] = [S, 1, L, 1]
    # repeat → [S, T, L, 1] → reshape to [S*T, L, 1]
    X = grid_t[:, None, :, None].repeat(1, T, 1, 1).reshape(S * T, L, 1)*3


    # --- Dummy target values [S, L, T] ---
    #Y_raw = torch.randn(S, L, T)
    Y_raw = torch.tensor([[[0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 1.],
                            [0., 0., 1., 0., 0.]],

                            [[0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 1.],
                            [1., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0.]],

                            [[0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 1., 0., 0.],
                            [1., 0., 0., 0., 0.]],

                            [[0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 1.],
                            [0., 0., 1., 0., 0.],
                            [1., 0., 0., 0., 0.]]])

    # Reshape Y to [S*T, L]
    Y = Y_raw.permute(0, 2, 1).reshape(S * T, L)

    # --- Likelihood ---
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([S * T]),
                                                         noise_prior = gpytorch.priors.NormalPrior(loc=0, scale=1e-4), 
                                                                           has_global_noise= False, 
                                                                           has_task_noise=True, 
                                                                           noise_constraint = gpytorch.constraints.Interval(1e-6,1e-5)
                                                                           )

    # --- GP Model ---
    class MultiSeqMultiTaskGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):

            super().__init__(train_x, train_y, likelihood)

            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([S * T]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([S * T]), ard_num_dims=1),
                batch_shape=torch.Size([S * T])
            )
            #import ipdb; ipdb.set_trace()
            task_means = torch.tensor([1., 0., 0., 0., 0.])
            batch_means = task_means.repeat(S)

            self.covar_module.base_kernel.lengthscale =  0.5
            self.mean_module.initialize(constant=batch_means)

        def forward(self, x):
            mean_x = self.mean_module(x)             # [S*T, L]
            covar_x = self.covar_module(x)           # [S*T, L, L]
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
    model = MultiSeqMultiTaskGP(X, Y, likelihood)

    # Test inputs: [20, 1]
    #x_test = torch.linspace(0, L - 1, 4).view(-1, 1)  # [20, 1]
    # Expand to [S*T, 20, 1]
    #x _test_batch = x_test.unsqueeze(0).expand(S * T, -1, -1)
    test_grid = torch.tensor(
        [[[0.0000, 0.3333, 0.6667, 1.0000]],
        [[0.0000, 0.3333, 0.6667, 1.0000]],
        [[0.0000, 0.3333, 0.6667, 1.0000]],
        [[0.0000, 0.3333, 0.6667, 1.0000]]]).squeeze(1)*3

    x_test_batch = test_grid[:, None, :, None].repeat(1, T, 1, 1).reshape(S*T, L, 1)




    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model(x_test_batch)               # MVN with shape [S*T, 20]
        mean = preds.mean.view(S, T, -1).permute(0, 2, 1)  # [S, 20, T]
        var = preds.variance.view(S, T, -1).permute(0, 2, 1)  # [S, 20, T]


