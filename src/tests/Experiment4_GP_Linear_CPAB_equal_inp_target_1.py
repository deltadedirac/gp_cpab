
import gc,sys
import pdb

pdb.set_trace()
from tqdm import tqdm

import os, mlflow, mlflow.pytorch

from transformation.gp_cpab import gp_cpab
from transformation.configManager import configManager
from extra.utilities import *
from extra.dataLoaderDiffeo import *
from extra.LossFunctionsAlternatives import LossFunctionsAlternatives
from extra.Automatic_Report import Generate_Automatic_Report
from transformation.libcpab.libcpab.pytorch.interpolation import interpolate1D


def training_theta_optima(path_preexist_model, theta_est, maxiter, optimizer, T,x1, ref_msa, modeflag, msa_num, loss_vals, loss_function, inverse):

 
    if os.path.isfile(path_preexist_model):
        print ("Loading Deformation Model")
        theta_est = torch.load(path_preexist_model)
    else:
        pb = tqdm(desc='Alignment of samples', unit='iters', total=maxiter)
        for i in range(maxiter):
            optimizer.zero_grad()
            #pdb.set_trace()

            if inverse == True:
                opt_est = -theta_est

            if T.interpolation_type == 'linear':
                x1_trans, _,  forw_per = T.spatial_transformation(x1, ref_msa, opt_est, modeflag)

                loss = loss_function(method = 'Soft_Label_KLD', input = x1_trans, target = ref_msa, forw_per=forw_per)

            else:
                #pdb.set_trace()
                x1_trans, sampled_data, forw_per = T.spatial_transformation(x1, ref_msa, opt_est, modeflag)

                loss = loss_function(method = 'JSD', input = x1_trans, target = ref_msa, forw_per=forw_per)


            loss_vals.append(loss.item())
            loss.backward()
            optimizer.step()


            gc.collect()
            torch.cuda.empty_cache()
            #torch.save(theta_est, path_preexist_model)



            pb.update()
            pb.set_postfix({'loss': str(loss.item())})

            msa_num.append(x1_trans.detach().numpy().squeeze())


        pb.close()
        shapes = ref_msa.squeeze().shape
        msa_num = np.vstack(msa_num).reshape(-1, shapes[0], shapes[1])


    return theta_est

def get_interpolated_data(data, T, outsize):
    data1=data.unsqueeze(0)
    grid = T.uniform_meshgrid(outsize).repeat(data1.shape[0],1,1)
    out = interpolate1D(data1,grid,outsize)
    return out.squeeze(0)


if __name__ == "__main__":
    pdb.set_trace()
    print(__package__)
    '''---------------------------------------------------------'''
    std = configManager("../configs/setup2.yaml")
    device = std.parserinfo('*/Device')
    modeflag = std.parserinfo('*/Modeflag')
    window_grid = std.parserinfo('*/Window_grid')
    channels = std.parserinfo('*/Channels')
    option = std.parserinfo('*/Option')


    path = std.parserinfo('*/PathOrig4')
    path_MSA_test = std.parserinfo('*/PathMSAref4t')#PathMSAref4t
    path_preexist_model = std.parserinfo('*/Path_Preexist_Modellinear_equal_inp_target3')
    path_preexist_modelGP = std.parserinfo('*/Path_Preexist_ModelGP4')

    pathX123 = std.parserinfo('*/Path_Automated_ReportFolder')
    indexlogolinear = '_LI_3aa9g_padd'
    indexlogoGP = '_GP_3aa9g_padd'
    indexoutputT = 'debugging_cpab_domains.txt' #'results_3aa_Inverse_direct.txt'

    pdb.set_trace()
    # Reference Information - how the alignment should looks like
    alignment, ref_msa, alphabets, c2i, i2c, i2i,seqchar = read_clustal_align_output(path_MSA_test)
    '''---------------------------------------------------------'''
    '''---------------------------------------------------------'''
    '''---------------------------------------------------------'''

    '''     temporary for the test given by wouter'''
    #ref_msa = ref_msa.float(); 
    #ref_msa[:,0]=ref_msa[:,1]=ref_msa[:,2]=torch.tensor([[0.25,0.25,0.25,0.25]])
    #ref_msa[:,3]=ref_msa[:,4]=torch.tensor([[0.25,0.25,0.25,0.25]])
    #ref_msa = ref_msa.float();ref_msa = torch.cat( (ref_msa, torch.tensor([[[0.5, 0.5]]]) ), 1)


    
    #ref_msa = ref_msa.float(); ref_msa[:,1]=torch.tensor([0.25, 0.25, 0.25, 0.25])
    #ref_msa = ref_msa.float(); ref_msa[:,2]=torch.tensor([0.25, 0.25, 0.25, 0.25])
    #ref_msa = ref_msa.float(); ref_msa[:,4]=torch.tensor([0.25, 0.25, 0.25, 0.25])
    #ref_msa = ref_msa.float(); ref_msa[:,5]=torch.tensor([0.25, 0.25, 0.25, 0.25])
    
    #ref_msa = ref_msa.float(); ref_msa[:,1]= ref_msa[:,2]= ref_msa[:,3]= ref_msa[:,4]= ref_msa[:,5]=torch.tensor([0.25, 0.25, 0.25, 0.25])
    #ref_msa = ref_msa.float(); ref_msa[:,7]= ref_msa[:,8]= ref_msa[:,9]= ref_msa[:,10]= ref_msa[:,11]=torch.tensor([0.25, 0.25, 0.25, 0.25])


    '''---------------------------------------------------------'''
    '''---------------------------------------------------------'''
    '''---------------------------------------------------------'''
    # Raw Sequences, to see if we can align the sequences somehow
    dataset_msa = datasetLoader(pathBLAT_data = path, alphabet = alphabets, enable_variable_length=True)
    pdb.set_trace()
    x1 = dataset_msa.prot_space
    padd_replication = ref_msa.shape[1] - x1.shape[1]
    if padd_replication!=0:
        padd = torch.ones(1,padd_replication,4)*0.25; x1 = torch.cat(( x1 ,padd),1)
    else:
        padd = torch.tensor([])

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
    ''' NEEDS TO BE IN ZERO_BOUNDARY IN FALSE FOR GETTING PROPER RESULTS IN ALIGNMENTS'''
    pdb.set_trace()
    T = gp_cpab(ndim, std, backend='pytorch', device=device, zero_boundary=False,
                                             volume_perservation=False, override=False)


    #T.set_solver_params(nstepsolver=80, numeric_grad=False)

    T.interpolation_type = 'linear'

    x1 = T.backend.to(x1.clone().detach(), device=device)
    ref_msa = T.backend.to(ref_msa.clone().detach(), device=device)
    #X = train_x = torch.linspace(start=0, end= x1.shape[1]-1,steps=x1.shape[1]).float()
    #X = torch.linspace(start=0, end= window_grid-1,steps=window_grid).float()

    ''' LIKELIHOOD DEFINITIONS:'''

    if option == 'multitask':
        #T.get_interpolation_inductive_points(x1, x1.float()) 
        pdb.set_trace()
        
        
        outsize = (ref_msa.shape[1], ref_msa.shape[2])
        padded_idx = [*range(x1.shape[1],ref_msa.shape[1])]
        non_padded_idx = set(range(0, outsize[0])) - set(padded_idx) 
        non_padded_idx = [*non_padded_idx]

        ''' TEMP SOLUTION FOR THE EXPERIMENT WITH THE MODIFICATIONS'''
        '''---------------------------------------------------------------------------'''
        padded_idx = non_padded_idx[(ref_msa.shape[1] - padd_replication):]; non_padded_idx =  non_padded_idx[:(ref_msa.shape[1] - padd_replication)]
        '''---------------------------------------------------------------------------'''

        padded_idx = [ padded_idx ]; non_padded_idx = [ non_padded_idx ]

        T.get_interpolation_inductive_points(x1, x1.float(), 
                                            outsize = outsize, 
                                            padded_idx = padded_idx, 
                                            non_padded_idx = non_padded_idx,
                                            #separation_size_grid = T.params.inc[0],
                                            padding_option = 'padding_weights'   )
                                            #padding_option = 'none'   )

        
        
    
    '''
    -------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------

    '''

    ''' FOR DIRECT TRANSFORM AVOIDING LOCAL MINIMA IN CPAB WITHOUT NUMERICAL CONSTRAINS'''
    #theta_ref = torch.autograd.Variable(torch.tensor([[0.3,-0.0011522,-0.0018782, -0.2456, 0.3125]], requires_grad=True), requires_grad=True)


    ''' FOR INVERSE TRANSFORM AVOIDING LOCAL MINIMA IN CPAB WITHOUT NUMERICAL CONSTRAINS'''
    #theta_ref = torch.autograd.Variable(torch.tensor([[ -0.4336, 0.3217, 0.0090, 0.4256,  -0.7682]], requires_grad=True), requires_grad=True)
    ''' FOR CASES 1,2,3,4 GAPS WITH INVERSE'''
    pdb.set_trace()
    theta_ref = torch.autograd.Variable(torch.tensor([[-0.3,0.0011522,0.0018782, 0.2456, -0.3125]], requires_grad=True), requires_grad=True)
    #theta_ref  = get_interpolated_data(theta_ref, T, (T.params.nC+1,T.params.nC+1))
    ''' FOR CASES 10 GAPS WITH INVERSE'''
    #theta_ref = torch.autograd.Variable(torch.tensor([[0.003,  -0.006, 0.003, 0.006,  -0.003]], requires_grad=True), requires_grad=True)
    ''' INITIALIZING THETA USING OPTIMAL THETA LINEAR CASE FOR USING IN GP APPROACH'''
    #theta_ref = torch.autograd.Variable(torch.tensor([[ -0.0626,  -0.3114, 0.3597,  -0.1838, 0.2374]], requires_grad=True), requires_grad=True)

    
    '''
    -------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------

    '''

    ##theta_ref = torch.autograd.Variable(T.identity(1, epsilon=1e-6), requires_grad=True)
    theta_est = torch.autograd.Variable(theta_ref.clone(), requires_grad=True)
    theta_est_GP = torch.autograd.Variable(theta_ref.clone(), requires_grad=True)
    
    '''
    theta_est = torch.autograd.Variable(T.identity(1, epsilon=1e-6), requires_grad=True)
    theta_est_GP = torch.autograd.Variable(T.identity(1, epsilon=1e-6), requires_grad=True)
    '''

    '''lr=0.01 is the best one so far for linear interpolation and gp'''
    lr = std.parserinfo('*/lr') 
    wd = std.parserinfo('*/weight_decay')
    maxiter = std.parserinfo('*/maxiter')

    optimizer = torch.optim.AdamW([theta_est], lr=lr) #, weight_decay=wd)
    optimizerGP = torch.optim.AdamW([theta_est_GP], lr=lr)


    loss_function = LossFunctionsAlternatives()
    loss_function.get_dictionaries_to_mask_data(c2i, i2c, i2i)
    #pdb.set_trace()
    msa_num = []; loss_vals =[]
    msa_numGP = []; loss_valsgp =[]

    inverse = True


    pdb.set_trace()

    '''
    T.interpolation_type = 'linear'
    theta_est = training_theta_optima(path_preexist_model, theta_est, maxiter, optimizer, T,x1, ref_msa, 
                                                modeflag, msa_num, loss_vals, loss_function, inverse = inverse)
    pdb.set_trace()

    x1_trans1, _, forw_per = T.spatial_transformation(x1, ref_msa, -theta_est, modeflag)
    x1_trans1_dir,_, forw_per = T.spatial_transformation(x1_trans1, ref_msa, theta_est, modeflag)
    loss1 = loss_function(method = 'Soft_Label_KLD', input = x1_trans1, target = ref_msa, forw_per=forw_per)
    '''
    pdb.set_trace()








    T.interpolation_type = 'GP'
    theta_est_GP = training_theta_optima(path_preexist_modelGP, theta_est_GP, maxiter, optimizerGP, T, x1, ref_msa, 
                                                modeflag, msa_numGP, loss_valsgp, loss_function, inverse = inverse)

    pdb.set_trace()
    x1_trans3, sampled_data3, forw_per = T.spatial_transformation(x1, ref_msa, -theta_est_GP, modeflag)

    pdb.set_trace()
    #T.setup='None'
    T.interpolation_type = 'linear'
    x1_trans3_dir, _, forw_per = T.spatial_transformation(x1_trans3, ref_msa, theta_est_GP, modeflag)

    loss3 = loss_function(method = 'Soft_Label_KLD', input = x1_trans3, target = ref_msa, forw_per=forw_per)

    print('\n\n\n Loss Function by using GPs = '+ str(loss3) )
    

    #pdb.set_trace()
    pdb.set_trace()



    print('done')


    