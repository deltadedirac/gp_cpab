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
        # Reshape Y: [batches, points, tasks] -> [batches, tasks, points]
        # Dimensions
        B, N, T = train_y.shape
        _, _, D = train_x.shape
        self.B, self.T, self.N, self.D = B, T, N, D
        
        # Flatten targets: [B, N, T] -> [B, T*N]
        #train_y_flat = train_y.permute(0, 2, 1).reshape(B, -1)
        #super().__init__(train_x, train_y_flat, likelihood)
        super().__init__(train_x, train_y, likelihood)

        # Convert mean_inits
        if isinstance(initialization, list):
            mean_inits = \
                torch.tensor(initialization, device=self.device)
        assert mean_inits.shape[0] == tasks

        mean_inits = mean_inits.unsqueeze(0).expand(B, -1)

        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([B, T]),  # [batches, tasks]
            device = torch.device(self.device)
        )

        # Use any kernel you want for input correlation
        self.covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.RBFKernel(
                                                        batch_shape=torch.Size([B, T]),
                                                        ard_num_dims=D
                                                    ),
                                        batch_shape=torch.Size([B, T]),
                                        device=torch.device(self.device)
        )

        #import pdb; pdb.set_trace()
        self.covar_module.base_kernel.lengthscale =  lengthscale

        # Apply custom initialization
        self.mean_module.initialize(constant= mean_inits)
        # Initialize means
        self.mean_module.constant.data = mean_inits.clone()
        """if isinstance(initialization,list):
            per_task_init = torch.tensor(initialization, device=self.device).repeat(self.batches, -1)
            self.mean_module.initialize( constant= per_task_init )
        else:
            self.mean_module.initialize(constant= initialization,
                                        device = torch.device(self.device)) 
        """


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
        # Input x shape: [B, N, D]
        # Expand: [B, T, N, D]
        x_expanded = x.unsqueeze(1).expand(-1, self.T, -1, -1)
        
        # Compute per-task distribution
        mean_x = self.mean_module(x_expanded)  # [B, T, N]
        covar_x = self.covar_module(x_expanded)  # [B, T, N, N]

        '''
        # Create diagonal covariance across tasks
        # We'll manually create the full block-diagonal covariance matrix
                # Create block-diagonal covariance
        covar_lazy = BlockDiagLazyTensor(covar_x)
        
        # Flatten mean to match covariance: [batches, tasks*points]
        mean_flat = mean_x.reshape(self.B, -1)
        
        return gpytorch.distributions.MultivariateNormal(mean_flat, covar_lazy)
        '''
                # Create independent MVNs per task
        mvns = []
        for t in range(self.T):
            # Ensure covar is positive definite
            try:
                covar_t = covar_x[:, t].evaluate_kernel()
            except RuntimeError:
                covar_t = covar_x[:, t].add_jitter(1e-4)
            
            mvns.append(gpytorch.distributions.MultivariateNormal(
                mean_x[:, t], 
                covar_t
            ))
        
        # Combine into diagonal multitask normal
        return gpytorch.distributions.MultitaskMultivariateNormal.from_independent_mvns(mvns)


    def __call__(self, x, **kwargs):
        # Bypass GPyTorch's default prediction strategy
        return self.forward(x)










class DebuggableDiagonalTaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_inits):
        # Verify input shapes
        print("\n[DEBUG] Initialization shapes:")
        print(f"train_x shape: {train_x.shape} (expected [B, N, D])")
        print(f"train_y shape: {train_y.shape} (expected [B, N, T])")
        
        # Dimensions
        B, N, T = train_y.shape
        _, _, D = train_x.shape
        self.B, self.T, self.N, self.D = B, T, N, D
        
        # Flatten targets: [B, N, T] -> [B, T*N]
        train_y_flat = train_y.permute(0, 2, 1).reshape(B, -1)
        print(f"train_y_flat shape: {train_y_flat.shape} (expected [B, T*N])")
        
        super().__init__(train_x, train_y_flat, likelihood)
        
        # Convert mean_inits
        if isinstance(mean_inits, list):
            mean_inits = torch.tensor(mean_inits, dtype=torch.float32)
        assert len(mean_inits) == T, f"Expected {T} tasks, got {len(mean_inits)}"
        
        # Mean module with shape checks
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([B, T])
        )
        
        # Kernel with shape checks
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([B, T]),
                ard_num_dims=D
            ),
            batch_shape=torch.Size([B, T])
        )
        
        # Initialize means
        init_tensor = mean_inits.unsqueeze(0).expand(B, -1)
        self.mean_module.initialize(constant=init_tensor)
        print(f"Mean constant shape: {self.mean_module.constant.shape} (expected [B, T])")
        
    def forward(self, x):
        # Input shape verification
        print("\n[DEBUG] Forward pass:")
        print(f"Input x shape: {x.shape} (expected [B, N, D])")
        
        # Expand input: [B, N, D] -> [B, T, N, D]
        x_expanded = x.unsqueeze(1).expand(-1, self.T, -1, -1)
        print(f"x_expanded shape: {x_expanded.shape} (expected [B, T, N, D])")
        
        # Compute mean
        mean_x = self.mean_module(x_expanded)
        print(f"mean_x shape: {mean_x.shape} (expected [B, T, N])")
        
        # Compute covariance
        covar_x = self.covar_module(x_expanded)
        print(f"covar_x shape: {covar_x.shape} (expected [B, T, N, N])")
        
        # Verify covariance is positive definite
        try:
            covar_x = covar_x.evaluate_kernel()
            print("Covariance evaluation successful")
        except RuntimeError as e:
            print(f"Covariance evaluation failed: {e}")
            # Add small jitter for numerical stability
            covar_x = covar_x.add_jitter(1e-4)
            print("Added jitter to covariance matrix")
        
        # Create block-diagonal covariance
        covar_lazy = BlockDiagLazyTensor(covar_x)
        print(f"BlockDiagLazyTensor shape: {covar_lazy.shape} (expected [B, T*N, T*N])")
        
        # Flatten mean
        mean_flat = mean_x.reshape(self.B, -1)
        print(f"mean_flat shape: {mean_flat.shape} (expected [B, T*N])")
        
        # Create distribution
        try:
            mvn = gpytorch.distributions.MultivariateNormal(mean_flat, covar_lazy)
            print("Successfully created MultivariateNormal")
            return mvn
        except RuntimeError as e:
            print(f"Error creating distribution: {e}")
            print("Shapes during error:")
            print(f"mean_flat: {mean_flat.shape}, covar_lazy: {covar_lazy.shape}")
            # Fallback to diagonal covariance
            print("Falling back to diagonal approximation")
            variances = covar_x.diagonal(dim1=-2, dim2=-1).reshape(self.B, -1)
            return gpytorch.distributions.MultivariateNormal(mean_flat, gpytorch.lazy.DiagLazyTensor(variances))
        
'''

if __name__ == "__main__":

    # Create dummy data with known shapes
    B, N, T, D = 4, 4, 5, 1  # Batches, Points, Tasks, Dimensions
    train_x = torch.randn(B, N, D)
    train_y = torch.randn(B, N, T)
    mean_inits = [1.0, 0.0, 0.0, 0.0, 0.0]

    # Initialize model with debug output
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DebuggableDiagonalTaskGP(train_x, train_y, likelihood, mean_inits)

    # Test forward pass
    print("\n[TEST] Running forward pass with training data:")
    with torch.no_grad():
        try:
            output = model(train_x)
            print("\nForward pass succeeded!")
            print(f"Output mean shape: {output.mean.shape}")
            print(f"Output covariance shape: {output.covariance_matrix.shape if hasattr(output, 'covariance_matrix') else 'LazyTensor'}")
        except Exception as e:
            print(f"\nForward pass failed: {e}")
            import traceback
            traceback.print_exc()

    # Test prediction
    test_x = torch.randn(B, 3, D)
    print("\n[TEST] Running prediction with test data:")
    model.eval()
    with torch.no_grad():
        try:
            test_output = model(test_x)
            print("\nPrediction succeeded!")
            print(f"Test mean shape: {test_output.mean.shape}")
            # Reshape to interpretable format
            test_mean = test_output.mean.reshape(B, T, -1)
            print(f"Reshaped mean: [B, T, P] = {test_mean.shape}")
        except Exception as e:
            print(f"\nPrediction failed: {e}")


'''