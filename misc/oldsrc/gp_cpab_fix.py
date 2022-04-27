#from calendar import c
import torch, gpytorch
import pdb, math, os, gc
from tqdm import tqdm

from utilities import *
from dataLoaderDiffeo import *
from configManager import *
from libcpab.libcpab import Cpab
#import gpytorch
from LossFunctionsAlternatives import LossFunctionsAlternatives
from gp_setup_fixed import BatchIndependentMultitaskGPModel
from functools import singledispatch


class gp_cpab(Cpab):

    def __init__(self, tess_size, config, backend = 'numpy',
                        device = 'cpu', zero_boundary = True, 
                        volume_perservation = False, override = False):
        super().__init__(tess_size, backend=backend, device=device, zero_boundary=zero_boundary, 
                                    volume_perservation=volume_perservation, override=override)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.config = config
        self.interpolation_type = config.parserinfo('*/Interpolation_type')
        self.tasks = config.parserinfo('*/Tasks')
        self.option = config.parserinfo('*/Option')

    def get_interpolation_inductive_points(self,X, multiout_Y, likelihood):
        #pdb.set_trace()
        self.X_GP = X
        self.outsize_orig_data=(X.shape[1],X.shape[2])
        self.multiout_Y_GP = multiout_Y
        if self.option == 'multitask':
            self.likelihood_Multitask = likelihood

    def predict_operation(self, test_points):
        self.multiout_GP_Interpolator.eval()
        self.likelihood_Multitask.eval()
        with gpytorch.settings.fast_pred_var():
            trans_data = self.likelihood_Multitask(self.multiout_GP_Interpolator(test_points))

        return trans_data

    
    def transform_data(self, data, theta, outsize):

        self._check_type(data); self._check_device(data)
        self._check_type(theta); self._check_device(theta)

        ''' Creation of Mesh Grid with same size as the reference output size'''
        self.grid = self.uniform_meshgrid(outsize)


        if self.interpolation_type == 'linear':
            grid_t = self.transform_grid(self.grid, theta)
            data_t = self.interpolate(data, grid_t, outsize)
            return data_t
        else:
            grid_no_expansion = self.uniform_meshgrid(self.outsize_orig_data)
            grid_t_no_expansion = self.transform_grid(grid_no_expansion, theta)
            data_t, sampled_data = self.interpolate_GP_1D_multitask2(data, grid_t_no_expansion, outsize)
            return data_t, sampled_data
    

    def interpolate_GP_1D_multitask2(self,data, grid_t_inv, outsize):
    # Problem size
        n_batch = data.shape[0]
        n_channels = data.shape[1]
        width = outsize[0]
        out_width = outsize[0]
        
        #pdb.set_trace()
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

        # Index
        data_sampled = data[batch_idx, :, x0]
        #pdb.set_trace()
        self.multiout_GP_Interpolator = BatchIndependentMultitaskGPModel(x, data_sampled, 
                                                                        self.likelihood_Multitask, self.config)

        trans_data = self.predict_operation(self.grid.flatten() * (width-1))
        trans_data_train = self.predict_operation(x)

        with gpytorch.settings.fast_pred_var():#pdb.set_trace()
            cc = trans_data.mean.flatten()
            ctrain = trans_data_train.mean.flatten()        
            self.ctrain = torch.reshape(ctrain, (n_batch, len(x), n_channels))

            self.lower,self.upper =  trans_data.confidence_region()
            self.lower_train,self.upper_train =  trans_data_train.confidence_region()
            sampled_data = trans_data.rsample().unsqueeze(0).flatten()

        gc.collect()
        torch.cuda.empty_cache()
         # Reshape
        new_data = torch.reshape(cc, (n_batch, out_width, n_channels))
        sampled_data = torch.reshape(sampled_data, (n_batch, out_width, n_channels))
        new_data = new_data.permute(0, 2, 1)
        sampled_data = sampled_data.permute(0, 2, 1)

        return new_data.contiguous(), sampled_data.contiguous()

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



def spatial_transformation2(T, raw_data, ref, theta, modeflag):
    # Transform the images
    forw,rev,outsize = determine_trans_dim2(raw_data,ref,modeflag)
    if T.interpolation_type == 'linear':
        trans_data = T.transform_data(raw_data.permute(forw), theta, outsize=outsize)
        trans_data = trans_data.permute(rev)
        return trans_data, forw
    else:
        trans_data, sampled_data = T.transform_data(raw_data.permute(forw), theta, outsize=outsize)
        trans_data = trans_data.permute(rev); sampled_data = sampled_data.permute(rev)
        return trans_data, sampled_data, forw

def distribution_sampling(modesampling,modeflag,x1_trans_mean,x1_trans_sample, msa_num):

    if modeflag == '2D':
        #if modesampling == True:
            Softmax=torch.nn.Softmax(dim=3) # take the axis 3 from (1,172,1,20) -> over 20 aas
            x1_trans = Softmax(x1_trans_mean)
            A = x1_trans_mean.argmax(-1).transpose(2,1).reshape(x1_trans_mean.argmax(-1).transpose(2,1).shape[2])
        #else:
            #A = x1_trans_sample.argmax(-1)
            msa_num.append(A)
    else:
        #if modesampling == True:
            if x1_trans_sample == None:
                Softmax=torch.nn.Softmax(dim=2) # take the axis 2 from (1,172,20) -> over 20 aas
                x1_trans_mean = Softmax(x1_trans_mean)
                A = x1_trans_mean.argmax(-1).reshape(x1_trans_mean.argmax(-1).shape[1])
            else:
                Softmax=torch.nn.Softmax(dim=2) # take the axis 2 from (1,172,20) -> over 20 aas
                x1_trans_sample = Softmax(x1_trans_sample)
                A = x1_trans_sample.argmax(-1).reshape(x1_trans_sample.argmax(-1).shape[1])
        #else:
            #A = x1_trans_sample.argmax(-1)
            msa_num.append(A)

'''Main Process'''
if __name__ == "__main__":

    '''---------------------------------------------------------'''
    """
    path = 'data/orig_3aa.fasta'
    path_MSA_test = 'data/ref_3aa.aln'
    device = 'cpu'
    modeflag = '1D'
    window_grid = 1100#178#174 #200 for 2D
    channels = 4
    option = 'multitask'
    path_preexist_model = 'models/CPABdeformGPB2.pth'
    path_preexist_likelihood = 'models/CPABdeform_likelihoodGPB2.pth'
    """
    std = configManager();pdb.set_trace()
    path = std.parserinfo('*/PathOrig')
    path_MSA_test = std.parserinfo('*/PathMSAref')
    device = std.parserinfo('*/Device')
    modeflag = std.parserinfo('*/Modeflag')
    window_grid = std.parserinfo('*/Window_grid')
    channels = std.parserinfo('*/Channels')
    option = std.parserinfo('*/Option')
    path_preexist_model = std.parserinfo('*/Path_Preexist_Model')
    path_preexist_likelihood = std.parserinfo('*/Path_Preexist_Likelihood')
    
    '''---------------------------------------------------------'''

    # Reference Information - how the alignment should looks like
    alignment, ref_msa, alphabets, c2i, i2c, i2i,seqchar = read_clustal_align_output(path_MSA_test)
    # Raw Sequences, to see if we can align the sequences somehow
    dataset_msa = datasetLoader(pathBLAT_data = path, alphabet = alphabets, enable_variable_length=True)
    x1 = dataset_msa.prot_space
    # reconvert the dimensions of reference and input datasets
    if modeflag == '2D':
      # convert into [#channels, #width, #height, #channels]
      x1 = x1.reshape(x1.shape[0], x1.shape[1], 1, x1.shape[2])
      ref_msa = ref_msa.reshape(ref_msa.shape[0], ref_msa.shape[1], 1, ref_msa.shape[2])
      ndim = [window_grid,1]
    else:
      ndim = [window_grid]

    '''Transformations for Reference Alignment'''
    '''-----------------------------------------------------------------------------------------------'''
    '''      JUST FOR CAPTURING DEFORMATION PATTERNS ALONG THE CPAB, HOW IT STARS AND IT ENDS:        '''
    '''-----------------------------------------------------------------------------------------------'''
    #T = Cpab(ndim, backend='pytorch', device=device, zero_boundary=True,
    #                                         volume_perservation=False, override=False)
    T = gp_cpab(ndim, std, backend='pytorch', device=device, zero_boundary=True,
                                             volume_perservation=False, override=False)
    #T.set_solver_params(nstepsolver=150)]
    pdb.set_trace()
    T.interpolation_type = 'linear'
    x1 = T.backend.to(x1, device=device)
    ref_msa = T.backend.to(ref_msa, device=device)
    #X = train_x = torch.linspace(start=0, end= x1.shape[1]-1,steps=x1.shape[1]).float()
    #X = torch.linspace(start=0, end= window_grid-1,steps=window_grid).float()

    ''' LIKELIHOOD DEFINITIONS:'''
    if option == 'multitask':
        constrain = std.parserinfo('*/noise_constraint')
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=channels, 
                                                                        noise_constraint=gpytorch.constraints.Interval(constrain[0],constrain[1]))#,       #,
                                                                    #has_global_noise=True, 
                                                                    #has_task_noise=False)
        
        T.get_interpolation_inductive_points(x1, x1.float(), likelihood)



    theta_ref = T.sample_transformation(ref_msa.shape[0])

    '''
    Experiment one: Alignment of Sequence by using Aligned Observational Sequences as reference'''
    '''-------------------------------------------------------------------------------------------'''
    
    theta_est = torch.autograd.Variable(T.identity(1, epsilon=1e-6), requires_grad=True)

    '''lr=0.01 is the best one so far for linear interpolation and gp'''
    lr = std.parserinfo('*/lr') 
    wd = std.parserinfo('*/weight_decay')
    maxiter = std.parserinfo('*/maxiter')

    optimizer = torch.optim.Adam([theta_est], lr=lr, weight_decay=wd) #=> FOR MSE

    pdb.set_trace()
    loss_function = LossFunctionsAlternatives()
    loss_function.get_dictionaries_to_mask_data(c2i, i2c, i2i)
    #pdb.set_trace()
    msa_num = []; loss_vals =[]

    if os.path.isfile(path_preexist_model):
        print ("Loading Deformation Model")
        theta_est = torch.load(path_preexist_model)
    else:
        pb = tqdm(desc='Alignment of samples', unit='iters', total=maxiter)
        for i in range(maxiter):
            optimizer.zero_grad()
            #pdb.set_trace()
            #x1_trans, forw_per, rever = spatial_transformation(T, x1, theta_est, modeflag)
            if T.interpolation_type == 'linear':
                x1_trans, forw_per = spatial_transformation2(T, x1, ref_msa, theta_est, modeflag)
                '''loss = loss_function(method = 'CEmask', input = x1_trans, target = ref_msa, forw_per=forw_per)'''
                loss = loss_function(method = 'CEmask', input = x1_trans, target = ref_msa, forw_per=forw_per)

            else:
                x1_trans, sampled_data, forw_per = spatial_transformation2(T, x1, ref_msa, theta_est, modeflag)
                '''loss = loss_function(method = 'CEmask', input = x1_trans, target = ref_msa, forw_per=forw_per)'''
                loss = loss_function(method = 'CEmask', input = sampled_data, target = ref_msa, forw_per=forw_per)

            loss_vals.append(loss.item())
            loss.backward()
            optimizer.step()


            gc.collect()
            torch.cuda.empty_cache()
            torch.save(theta_est, path_preexist_model)


            '''
            create_deformation_movie(T.u_grid.detach().numpy().flatten(), T.grid_t.detach().numpy(), 'grid_evo/linear/'+str(i)+'.png', loss.item(), i)
            
            torch.save(theta_est, path_preexist_model)
            '''

            pb.update()
            pb.set_postfix({'loss': str(loss.item())})
            '''
            if modeflag == '2D':
                Softmax=torch.nn.Softmax(dim=3) # take the axis 3 from (1,172,1,20) -> over 20 aas
                x1_trans = Softmax(x1_trans)
                msa_num.append(x1_trans.argmax(-1).transpose(2,1).reshape(x1_trans.argmax(-1).transpose(2,1).shape[2]))
            else:
                Softmax=torch.nn.Softmax(dim=2) # take the axis 2 from (1,172,20) -> over 20 aas
                x1_trans = Softmax(x1_trans)
                msa_num.append(x1_trans.argmax(-1).reshape(x1_trans.argmax(-1).shape[1]))
            
            '''
            if T.interpolation_type == 'linear':
                distribution_sampling(False,modeflag,x1_trans, None, msa_num)
            else:
                distribution_sampling(False,modeflag,x1_trans, None, msa_num)
            

        pb.close()



    '''
        ---------------------------------------------------------------
        ---------------------------------------------------------------
        Testing the transformation by using GPs as interpolator
        ---------------------------------------------------------------
        ---------------------------------------------------------------
    '''
    pdb.set_trace()
    T.interpolation_type = 'linear'
    x1_trans1, forw_per = spatial_transformation2(T, x1, ref_msa, theta_est, modeflag)
    loss1 = loss_function(method = 'CE', input = x1_trans1, target = ref_msa, forw_per=forw_per)
    print('\n\n\n Loss Function by using GPs = '+ str(loss1) )

    T.interpolation_type = 'GP'
    x1_trans2, sampled_data, forw_per = spatial_transformation2(T, x1, ref_msa, theta_est, modeflag)
    loss2 = loss_function(method = 'CE', input = x1_trans2, target = ref_msa, forw_per=forw_per)
    print('\n\n\n Loss Function by using GPs = '+ str(loss2) )
    pdb.set_trace()

    #loss_vals.append(loss.item())
    '''
            ---------------------------------------------------------------
            ---------------------------------------------------------------
            Plotting the logos for RAW, ALIGNED (REF) and results with CPAB
            ---------------------------------------------------------------
            ---------------------------------------------------------------
    '''

    if loss_vals: 
        plot_logos_results(x1,ref_msa,x1_trans1, alphabets, c2i, i2c, i2i, msa_num, '_linear_',loss_vals, modeflag)
        pdb.set_trace()
        
        plot_logos_results(x1,ref_msa,x1_trans2, alphabets, c2i, i2c, i2i, msa_num, '_GP4_',loss_vals, modeflag)
        pdb.set_trace()
        x1_trans2_train = T.ctrain

    plot_GP_components3(T.ctrain,T,channels,'fixedGP_res/GP4comp_train_', scalefactor=10, mode = 'see_train')
    '''plot_GP_components3(x1_trans2_train,T,channels,'fixedGP_res/GP4comp_train_', scalefactor=10, mode = 'see_train')'''
    #plot_GP_components3(x1_trans2,T,channels,'fixedGP_res/GP4comp_', scalefactor=10, mode = 'see_test')

    print('done')



