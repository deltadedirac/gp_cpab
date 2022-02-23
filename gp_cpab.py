import pdb, torch, math
import os

import gc
from tqdm import tqdm

from Bio.SeqIO.FastaIO import SimpleFastaParser
from dataLoaderDiffeo import *
#from torch.utils.data import Dataset, DataLoader
from libcpab.libcpab import Cpab
#import gpytorch
from LossFunctionsAlternatives import LossFunctionsAlternatives

import gpytorch
from gp_interpolation import GPInterpolation, BatchIndependentMultitaskGPModel, BatchesGPModel
from utilities import *

class gp_cpab(Cpab):

    def __init__(self, tess_size, backend='numpy', device='cpu', zero_boundary=True, volume_perservation=False, override=False,interpolation_type='linear'):
        super().__init__(tess_size, backend=backend, device=device, zero_boundary=zero_boundary, volume_perservation=volume_perservation, override=override)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #self.interpolator = GPInterpolation(torch.tensor([0.]), torch.tensor([0.]), self.likelihood)
        #self.interpolator2 = GPInterpolation(torch.tensor([0.]), torch.tensor([0.]), self.likelihood)

        self.interpolation_type = interpolation_type
        self.cpabgrid = torch.tensor([])
        #self.theta_est = torch.autograd.Variable(self.identity(1, epsilon=1e-6), requires_grad=True)

    def get_interpolation_inductive_points(self,X, multiout_Y, likelihood, tasks, option='multitask'):
        #pdb.set_trace()
        self.X_GP = X
        self.multiout_Y_GP = multiout_Y
        self.tasks = tasks
        self.option = option
        if option == 'multitask':
            self.likelihood_Multitask = likelihood
            #self.multiout_GP_Interpolator = BatchIndependentMultitaskGPModel(self.X_GP, self.multiout_Y_GP, self.likelihood_Multitask, self.tasks)
        else:
            self.likelihood_batches = likelihood
            tmp = self.X_GP.view(1, -1, 1).repeat(self.tasks, 1, 1)
            #self.batches_GP_Interpolator = BatchesGPModel(tmp, self.multiout_Y_GP, self.likelihood_batches, self.tasks)



    def transform_data(self, data, theta, outsize):
        """ Combination of the transform_grid and interpolate methods for easy
            transformation of data.
        Arguments:
            data: [n_batch, *data_shape] tensor, with input data. The format of
                the data_shape depends on the dimension of the data AND the
                backend that is being used. In tensorflow and numpy:
                    In 1D: [n_batch, number_of_features, n_channels]
                    In 2D: [n_batch, width, height, n_channels]
                    In 3D: [n_batch, width, height, depth, n_channels]
                In pytorch:
                    In 1D: [n_batch, n_channels, number_of_features]
                    In 2D: [n_batch, n_channels, width, height]
                    In 3D: [n_batch, n_channels, width, height, depth]
            theta: [n_batch, d] matrix with transformation parameters. Each row
                correspond to a transformation.
            outsize: list, number of points in each direction that is transformed
                and interpolated
        Output:
            data_t: [n_batch, *outsize] tensor, transformed and interpolated data
        """

        self._check_type(data); self._check_device(data)
        self._check_type(theta); self._check_device(theta)
        #pdb.set_trace()

        grid = self.uniform_meshgrid(outsize)
        self.u_grid = grid
        self.grid_t = self.transform_grid(grid, theta)
        #data_t = self.interpolate(data, grid_t, outsize)
        if self.interpolation_type == 'linear':
            data_t = self.interpolate(data, self.grid_t, outsize)
        else:
            #data_t = self.interpolate_GP_1D_multitask(data, grid_t, outsize)
            '''data_t = self.interpolate_GP_1D_group_gps(data, grid_t, outsize)'''
            data_t = self.interpolate_GP_1D_multitask(data, self.grid_t, outsize)

        return data_t

    def predict_operation(self, test_points):
        if self.option == 'multitask':
            self.multiout_GP_Interpolator.eval()
            self.likelihood_Multitask.eval()
            with gpytorch.settings.fast_pred_var():
                trans_data = self.likelihood_Multitask(self.multiout_GP_Interpolator(test_points))

        else: 
            self.batches_GP_Interpolator.eval()
            self.likelihood_batches.eval()
            with gpytorch.settings.fast_pred_var():
                trans_data = self.likelihood_batches(self.batches_GP_Interpolator(test_points))

        return trans_data

    def interpolate_GP_1D_multitask(self,data, grid, outsize):
    # Problem size
        n_batch = data.shape[0]
        n_channels = data.shape[1]
        width = data.shape[2]
        out_width = outsize[0]
        
        #pdb.set_trace()
        x = grid[:,0].flatten()
        self.cpabgrid = grid[:,0].flatten() * (width-1)
        # Scale to domain
        x = x * (width-1)
        # Do sampling
        x0 = torch.floor(x).to(torch.int64)
        # Clip values
        x0 = torch.clamp(x0, 0, width-1)


        self.ugrid = self.u_grid.flatten(); 
        self.ugrid = self.ugrid * (width-1)

        # Batch effect
        batch_size = out_width
        batch_idx = torch.arange(n_batch).repeat(batch_size, 1).t().flatten()

        #pdb.set_trace()
        # Index
        data_sampled = data[batch_idx, :, x0]
        #pdb.set_trace()
        self.multiout_GP_Interpolator = BatchIndependentMultitaskGPModel(x, 
                                                                        data_sampled, 
                                                                        self.likelihood_Multitask, self.tasks)

        trans_data = self.predict_operation(self.ugrid.flatten())
        trans_data_train = self.predict_operation(self.cpabgrid.flatten())

        #pdb.set_trace()
        c = trans_data.mean.flatten()
        ctrain = trans_data_train.mean.flatten()

        self.lower,self.upper =  trans_data.confidence_region()
        self.lower_train,self.upper_train =  trans_data_train.confidence_region()

        gc.collect()
        torch.cuda.empty_cache()
         # Reshape
        new_data = torch.reshape(c, (n_batch, out_width, n_channels))
        new_data = new_data.permute(0, 2, 1)

        '''JUST FOR SEEING WHAT IS GOING ON WITH MY TRAIN POINTS'''
        #new_data.requires_grad
        self.ctrain = torch.reshape(ctrain, (n_batch, out_width, n_channels))
        return new_data.contiguous()




def determine_trans_dim(raw_data,modeflag):
    '''
    In pytorch:
        In 1D: [n_batch, n_channels, number_of_features]
        In 2D: [n_batch, n_channels, width, height]
    '''
    forw=[];rev=[];outsize=()
    if modeflag=='1D':
      forw=[0,2,1]; rev=[0,2,1]
      outsize=(raw_data.shape[1],raw_data.shape[2])
    else:
      forw=[0,3,1,2]; rev=[0,2,3,1] #=> height, width
      #forw=[0,3,2,1]; rev=[0,2,3,1]
      outsize=(raw_data.shape[1],raw_data.shape[2])
    return forw,rev,outsize

def determine_trans_dim2(raw_data, ref, modeflag):
    '''
    In pytorch:
        In 1D: [n_batch, n_channels, number_of_features]
        In 2D: [n_batch, n_channels, width, height]
    '''
    forw,rev,outsize = determine_trans_dim(raw_data,modeflag)
    outsize=(ref.shape[1],ref.shape[2])
    return forw,rev,outsize


def spatial_transformation(T,raw_data,theta, modeflag):
    # Transform the images
    forw,rev,outsize = determine_trans_dim(raw_data,modeflag)
    trans_data = T.transform_data(raw_data.permute(forw), theta, outsize=outsize)
    trans_data = trans_data.permute(rev)

    return trans_data, forw

def spatial_transformation2(T, raw_data, ref, theta, modeflag):
    # Transform the images
    forw,rev,outsize = determine_trans_dim2(raw_data,ref,modeflag)
    trans_data = T.transform_data(raw_data.permute(forw), theta, outsize=outsize)
    trans_data = trans_data.permute(rev)
    return trans_data, forw

    

'''Main Process'''
if __name__ == "__main__":

    '''
    path = 'BLAT1_without_lowers.fasta'
    path_MSA_test = '1_Align_seq_BLAT.aln'
    device = 'cpu'
    modeflag = '1D'
    window_grid = 181#178#174 #200 for 2D
    channels = 20
    option = 'multitask'
    '''
    '''---------------------------------------------------------'''
    path = 'data/orig_3aa.fasta'
    path_MSA_test = 'data/ref_3aa.aln'
    device = 'cpu'
    modeflag = '1D'
    window_grid = 1400#178#174 #200 for 2D
    channels = 4
    option = 'multitask'
    path_preexist_model = 'models/CPABdeformGPB2.pth'
    path_preexist_likelihood = 'models/CPABdeform_likelihoodGPB2.pth'
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
    T = gp_cpab(ndim, backend='pytorch', device=device, zero_boundary=True,
                                             volume_perservation=False, override=False, interpolation_type = 'GP')
    #T.set_solver_params(nstepsolver=150)]
    pdb.set_trace()
    T.interpolation_type = 'GP'
    x1 = T.backend.to(x1, device=device)
    ref_msa = T.backend.to(ref_msa, device=device)
    #X = train_x = torch.linspace(start=0, end= x1.shape[1]-1,steps=x1.shape[1]).float()
    X = train_x = torch.linspace(start=0, end= window_grid-1,steps=window_grid).float()

    ''' LIKELIHOOD DEFINITIONS:'''
    if option == 'multitask':
        
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=channels, 
                                                                        noise_constraint=gpytorch.constraints.Interval(0,1e-4))#,       #,
                                                                    #has_global_noise=True, 
                                                                    #has_task_noise=False)
        
        T.get_interpolation_inductive_points(X, x1.float(), likelihood, channels, option = 'multitask')

    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([channels]))
        T.get_interpolation_inductive_points(X, x1.float(), likelihood, channels, option = 'batches')


    theta_ref = T.sample_transformation(ref_msa.shape[0])

    '''
    Experiment one: Alignment of Sequence by using Aligned Observational Sequences as reference'''
    '''-------------------------------------------------------------------------------------------'''
    
    theta_est = torch.autograd.Variable(T.identity(1, epsilon=1e-6), requires_grad=True)

    '''lr=0.01 is the best one so far for linear interpolation and gp'''
    lr = 0.01
    #optimizer = torch.optim.SGD([theta_est], lr=lr, weight_decay=wd)#, weight_decay=1e-7) #=> FOR MSE
    pdb.set_trace()
    optimizer = torch.optim.Adam([theta_est], lr=lr, weight_decay=1e-4) #=> FOR MSE

    pdb.set_trace()
    maxiter = 1000 
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
            x1_trans, forw_per = spatial_transformation2(T, x1, ref_msa, theta_est, modeflag)
            loss = loss_function(method = 'CEmask', input = x1_trans, target = ref_msa, forw_per=forw_per)
            loss_vals.append(loss.item())
            loss.backward()
            optimizer.step()


            gc.collect()
            torch.cuda.empty_cache()
            #pdb.set_trace()

            '''
            create_deformation_movie(T.u_grid.detach().numpy().flatten(), T.grid_t.detach().numpy(), 'grid_evo/linear/'+str(i)+'.png', loss.item(), i)
            '''
            #torch.save(theta_est, path_preexist_model)
            

            pb.update()
            pb.set_postfix({'loss': str(loss.item())})
            if modeflag == '2D':
                Softmax=torch.nn.Softmax(dim=3) # take the axis 3 from (1,172,1,20) -> over 20 aas
                x1_trans = Softmax(x1_trans)
                msa_num.append(x1_trans.argmax(-1).transpose(2,1).reshape(x1_trans.argmax(-1).transpose(2,1).shape[2]))
            else:
                Softmax=torch.nn.Softmax(dim=2) # take the axis 2 from (1,172,20) -> over 20 aas
                x1_trans = Softmax(x1_trans)
                msa_num.append(x1_trans.argmax(-1).reshape(x1_trans.argmax(-1).shape[1]))
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
    x1_trans2, forw_per = spatial_transformation2(T, x1, ref_msa, theta_est, modeflag)
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



