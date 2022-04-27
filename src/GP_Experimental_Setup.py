from gp_cpab import *
import mlflow, mlflow.pytorch
import pdb
from Automatic_Report import *
from LossFunctionsAlternatives import LossFunctionsAlternatives


def training_theta_optima(path_preexist_model, theta_est, maxiter, optimizer, T,x1, ref_msa, modeflag, msa_num, loss_vals, loss_function):

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
                x1_trans, forw_per = T.spatial_transformation(x1, ref_msa, theta_est, modeflag)
                '''loss = loss_function(method = 'CEmask', input = x1_trans, target = ref_msa, forw_per=forw_per)'''
                #pdb.set_trace()
                loss = loss_function(method = 'CEmask', input = x1_trans, target = ref_msa, forw_per=forw_per)

            else:
                x1_trans, sampled_data, forw_per = T.spatial_transformation(x1, ref_msa, theta_est, modeflag)
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
            '''

            pb.update()
            pb.set_postfix({'loss': str(loss.item())})
 
            if T.interpolation_type == 'linear':
                distribution_sampling(False,modeflag,x1_trans, None, msa_num)
            else:
                distribution_sampling(False,modeflag,x1_trans, None, msa_num)
            

        pb.close()

    return theta_est


if __name__ == "__main__":

    '''---------------------------------------------------------'''
    std = configManager("../configs/setup2.yaml"); pdb.set_trace()
    path = std.parserinfo('*/PathOrig')
    device = std.parserinfo('*/Device')
    modeflag = std.parserinfo('*/Modeflag')
    window_grid = std.parserinfo('*/Window_grid')
    channels = std.parserinfo('*/Channels')
    option = std.parserinfo('*/Option')

    path_MSA_test = std.parserinfo('*/PathMSAref3')
    path_preexist_model = std.parserinfo('*/Path_Preexist_Modellinear3')
    path_preexist_modelGP = std.parserinfo('*/Path_Preexist_ModelGP3')

    pathX123 = std.parserinfo('*/Path_Automated_ReportFolder')
    indexlogolinear = '_LI_3aa2g_'
    indexlogoGP = '_GP_3aa2g_'
    indexoutputT = 'results_3aa3g_3_c.txt'
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

    pdb.set_trace()
    #T.set_solver_params(nstepsolver=80, numeric_grad=False)

    T.interpolation_type = 'linear'

    x1 = T.backend.to(x1, device=device)
    ref_msa = T.backend.to(ref_msa, device=device)
    #X = train_x = torch.linspace(start=0, end= x1.shape[1]-1,steps=x1.shape[1]).float()
    #X = torch.linspace(start=0, end= window_grid-1,steps=window_grid).float()

    ''' LIKELIHOOD DEFINITIONS:'''
    if option == 'multitask':        
        T.get_interpolation_inductive_points(x1, x1.float())

    '''
    -------------------------------------------------------------------------------------------
    Experiment one: Alignment of Sequence by using Aligned Observational Sequences as reference
    -------------------------------------------------------------------------------------------
    '''
    pdb.set_trace()
    theta_ref = T.sample_transformation_with_prior(ref_msa.shape[0],length_scale=0.1)
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

    optimizer = torch.optim.Adam([theta_est], lr=lr) #, weight_decay=wd)
    optimizerGP = torch.optim.Adam([theta_est_GP], lr=lr)


    loss_function = LossFunctionsAlternatives()
    loss_function.get_dictionaries_to_mask_data(c2i, i2c, i2i)
    #pdb.set_trace()
    msa_num = []; loss_vals =[]
    msa_numGP = []; loss_valsgp =[]

    pdb.set_trace()
    T.interpolation_type = 'linear'
    theta_est = training_theta_optima(path_preexist_model, theta_est, maxiter, optimizer, T,x1, ref_msa, modeflag, msa_num, loss_vals, loss_function)
    x1_trans1, forw_per = T.spatial_transformation(x1, ref_msa, theta_est, modeflag)
    loss1 = loss_function(method = 'CE', input = x1_trans1, target = ref_msa, forw_per=forw_per)
    print('\n\n\n Loss Function by using GPs = '+ str(loss1) )

    T.interpolation_type = 'GP'
    theta_est_GP = training_theta_optima(path_preexist_modelGP, theta_est_GP, maxiter, optimizerGP, T, x1, ref_msa, modeflag, msa_numGP, loss_valsgp, loss_function)
    x1_trans2, sampled_data, forw_per = T.spatial_transformation(x1, ref_msa, theta_est, modeflag)
    loss2 = loss_function(method = 'CE', input = x1_trans2, target = ref_msa, forw_per=forw_per)
    print('\n\n\n Loss Function by using GPs = '+ str(loss2) )

    x1_trans3, sampled_data2, forw_per = T.spatial_transformation(x1, ref_msa, theta_est_GP, modeflag)
    loss2 = loss_function(method = 'CE', input = sampled_data, target = ref_msa, forw_per=forw_per)

    #pdb.set_trace()
    pdb.set_trace()
    Generate_Automatic_Report(x1, T, ref_msa, theta_est, theta_est_GP, msa_num, msa_numGP, 
                                alphabets, c2i, i2c, i2i, path, std, modeflag, 
                                x1_trans1, x1_trans2, x1_trans3, loss_vals, loss_valsgp,
                                pathX123, indexoutputT, indexlogolinear, indexlogoGP)

    '''
            ---------------------------------------------------------------
            ---------------------------------------------------------------
            Plotting the logos for RAW, ALIGNED (REF) and results with CPAB
            ---------------------------------------------------------------
            ---------------------------------------------------------------
    
    pdb.set_trace()
    if loss_vals: 
        plot_logos_results(x1,ref_msa,x1_trans1, alphabets, c2i, i2c, i2i, msa_num, '_linear_',loss_vals, modeflag)

    if loss_valsgp: 
        plot_logos_results(x1,ref_msa,x1_trans3, alphabets, c2i, i2c, i2i, msa_numGP, '_GP4_',loss_valsgp, modeflag)
        pdb.set_trace()
    '''
    #plot_GP_components3(T.ctrain,T,channels,'fixedGP_res/GP4comp_train_gap9', scalefactor=10, mode = 'see_train')

    '''
    mlflow.set_tracking_uri('http://127.0.0.1:5000')  # set up connection
    mlflow.set_experiment('test-experiment')          # set the experiment

    mlflow.log_metric('Final output linear', x1_trans1)
    mlflow.log_metric('Final output GP', x1_trans2)
    mlflow.log_metric('Raw data', x1)
    mlflow.log_metric('Reference', ref_msa)

    '''

    print('done')


    