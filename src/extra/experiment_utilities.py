

import pandas as pd
from .logomakers import logomaker_plots 
import torch
import torch.nn.functional as F
from .dataLoaderDiffeo import *
from .utilities import *
from .LossFunctionsAlternatives import LossFunctionsAlternatives
from ..transformation.libcpab.libcpab.pytorch.interpolation import interpolate1D

device='cpu'

def padding_pattern_input_target(x1, ref_msa, token, alphabet_dict):

    padd_replication = abs(ref_msa.shape[1] - x1.shape[1])
    padd = torch.tensor([])

    if padd_replication!=0:
        if token == 'gap':
            channel_factor = 1./ref_msa.shape[2]
            padd = torch.ones(1,padd_replication,ref_msa.shape[2])*channel_factor
        elif token == 'token':
            if '-' in alphabet_dict: token_number, num_class = alphabet_dict['-'], len(alphabet_dict)
            # in case there not exist the token, we have to append it as column to 
            # the target, because we are including the new symbol in the dictionary
            else: token_number, num_class = len(alphabet_dict),len(alphabet_dict)+1; ref_msa = torch.cat( (ref_msa,torch.zeros(ref_msa.shape[0],ref_msa.shape[1],1)), 2 )

            padd = F.one_hot(torch.tensor(token_number), num_class).repeat(1,padd_replication,1)#.repeat(1,padd_replication,len(alphabet_dict))
        
        # conditional where input is filled with padding to
        # equal target length or backwards depending of their difference
        if x1.shape[1] <= ref_msa.shape[1]: x1 = torch.cat(( x1 ,padd),1)
        else: ref_msa = torch.cat(( ref_msa ,padd),1)
        

    outsize = (ref_msa.shape[1], ref_msa.shape[2])
    padded_idx = [*range(x1.shape[1],ref_msa.shape[1])]
    non_padded_idx = set(range(0, outsize[0])) - set(padded_idx) 
    non_padded_idx = [*non_padded_idx]

    ''' TEMP SOLUTION FOR THE EXPERIMENT WITH THE MODIFICATIONS'''
    '''---------------------------------------------------------------------------'''
    padded_idx = non_padded_idx[(ref_msa.shape[1] - padd_replication):]; non_padded_idx =  non_padded_idx[:(ref_msa.shape[1] - padd_replication)]
    '''---------------------------------------------------------------------------'''

    padded_idx = [ padded_idx ]; non_padded_idx = [ non_padded_idx ]
    return padded_idx, non_padded_idx, x1, ref_msa, outsize


def fill_seqs_with_symbols(seq, padd_replication, c2i, symbol='gap'):
    if symbol == 'gap':
        channel_factor = 1./(seq.shape[2]-1) #1./seq.shape[2]
        padd = torch.ones(1,padd_replication,seq.shape[2]) * channel_factor
        padd[:,:,0]=0
    elif symbol == 'token':
        classes = len(c2i.keys())
        token = F.one_hot(torch.tensor(c2i['-']), classes)
        padd = token.repeat(1,padd_replication,1)
    elif symbol == 'token_target':
        classes = len(c2i.keys())
        token = F.one_hot(torch.tensor(c2i['.']), classes)
        padd = token.repeat(1,padd_replication,1)

    return torch.cat(( seq ,padd),1)

def replace_target_token_gaps(seq, c2i, token_sym):
    token_val = c2i[token_sym]
    val_gap = torch.ones(seq.shape[2])*1./(seq.shape[2]-1)
    val_gap[0]=0.0
    seq[seq.argmax(-1)==token_val] =  val_gap
    return seq

def padding_strategy_input_target(x1, ref_msa, token_x, token_target, c2i, after_bound = 4):

    len_inp = x1.shape[1]
    len_target = ref_msa.shape[1]
    padd_replication = ref_msa.shape[1] - x1.shape[1]

    # make input and target to same size
    if padd_replication > 0:
        x1 = fill_seqs_with_symbols(x1, abs(padd_replication), c2i, symbol=token_x)
        outsize = len_target + after_bound
    elif padd_replication < 0:
        ref_msa = fill_seqs_with_symbols(ref_msa, abs(padd_replication), c2i, symbol=token_target)
        outsize = len_inp + after_bound
    else: outsize = len_target

    # increase input and target seq size after the matching of both bound sizes
    if '-' in c2i: 
        x1 = fill_seqs_with_symbols(x1, after_bound, c2i, symbol=token_x)
        if token_target != 'token_target':
            ref_msa = replace_target_token_gaps(ref_msa, c2i, '.')
        ref_msa = fill_seqs_with_symbols(ref_msa, after_bound, c2i, symbol=token_target)    

    return x1, ref_msa, (outsize,outsize)

'''
# Reference Information - how the alignment should looks like
alph = ['-','.', 'L', 'Q', 'R']
alignment, ref_msa, alphabets, c2i, i2c, i2i,seqchar = read_clustal_align_output(path_MSA_test, alphabet=alph)
# Raw Sequences, to see if we can align the sequences somehow
dataset_msa = datasetLoader(pathBLAT_data = path, alphabet = alphabets, enable_variable_length=True)

x1 = dataset_msa.prot_space
fill_inp = 'token'
fill_target = 'token_target' #'gap'

x1, ref_msa, outsize = padding_strategy_input_target(x1.float(), ref_msa.float(), fill_inp, fill_target, c2i, after_bound = 5)
print('final input: \n {0} \n'.format(x1))
print('final target: \n {0} \n'.format(ref_msa))
'''


#def plot_logos_probs(x1_trans, alphabets, folderpath = pathfolder, name = path):  
def plot_logos_probs(x1_trans, alphabets, **kargs): #folderpath = pathfolder, name = path):  
    #from logomakers import logomaker_plots  


    alphabets_logo = [ i if i!='-' else 'X' for i in alphabets] 
    x1_trans_logo_input = df_construction_aas([ x1_trans.detach().numpy() ], x1_trans.shape, alphabets_logo)
    best = logomaker_plots.plotlogos(x1_trans_logo_input[x1_trans_logo_input.columns.tolist()] )

    return best

def get_interpolated_data(data, T, outsize):
    data1=data.unsqueeze(0)
    grid = T.uniform_meshgrid(outsize).repeat(data1.shape[0],1,1)
    out = interpolate1D(data1,grid,outsize)
    return out.squeeze(0)

def define_data_and_transformations(path_MSA_test, path, gp_params, option, T, type_of_fill='token,gap', length_filling = 3, **kargs):

    if 'alphabet' in kargs:
        dict_aas = kargs['alphabet']
    # Reference Information - how the alignment should looks like
    alignment, ref_msa, alphabets, c2i, i2c, i2i,seqchar = read_clustal_align_output(path_MSA_test, alphabet = dict_aas)
    # Raw Sequences, to see if we can align the sequences somehow
    dataset_msa = datasetLoader(pathBLAT_data = path, alphabet = alphabets, enable_variable_length=True)

    x1 = dataset_msa.prot_space
    #padded_idx, non_padded_idx, x1, ref_msa, outsize = padding_pattern_input_target(x1, ref_msa, type_of_fill, c2i)

    x1 = T.backend.to(x1.clone().detach(), device=device)
    ref_msa = T.backend.to(ref_msa.clone().detach(), device=device)

    fill_inp , fill_target = type_of_fill.split(',')
    x1, ref_msa, outsize = padding_strategy_input_target(x1.float(), ref_msa.float(), fill_inp, fill_target, c2i, after_bound = length_filling)
    padded_idx, non_padded_idx = [[]], [[]]

    
    return alphabets, c2i, i2c, i2i, dataset_msa, padded_idx, non_padded_idx, x1, ref_msa, outsize, T


def optimization_setup(conf, theta_est, theta_est_GP, c2i, i2c, i2i, keys = ['lr','weight_decay','maxiter']):
    '''lr=0.01 is the best one so far for linear interpolation and gp'''
        
    lr, wd, maxiter = conf.get_config_vals(keys)
    #lr = std.parserinfo('*/lr') 
    #wd = std.parserinfo('*/weight_decay')
    #maxiter = std.parserinfo('*/maxiter')

    optimizer = torch.optim.AdamW([theta_est], lr=lr) #, weight_decay=wd)
    optimizerGP = torch.optim.AdamW([theta_est_GP], lr=lr)


    loss_function = LossFunctionsAlternatives()
    loss_function.get_dictionaries_to_mask_data(c2i, i2c, i2i)

    return lr, wd, maxiter, optimizer, optimizerGP, loss_function

def plot_irregular_table(cm, x_lab, y_lab,cmap='viridis'): 
    import itertools 

    figure, ax = plt.subplots(figsize=(8,10)) # plt.figure(figsize=(15,15))
    if cmap=='none':
        im = ax.imshow(cm, interpolation='None') 
    else:
        im = ax.imshow(cm, interpolation='None', cmap='viridis') 
    ax.set_title("value matrix") 
    #ax.set_colorbar() 
    tick_marks_x = np.arange(0.5,len(x_lab)+0.5) 
    tick_marks_y = np.arange(0.5,len(y_lab)+0.5) 

    ax.set_xticks(tick_marks_x, x_lab, rotation=45) 
    ax.set_yticks(tick_marks_y, y_lab)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):   
        #color = "white" if cm[i, j] > threshold else "black"
        color = "white"
        ax.text(j, i, cm[i, j],  ha="center", va="center", color="w")  
    
    ax.set_ylabel('True label') 
    ax.set_xlabel('Predicted label') 
    
    ax.grid()
    plt.close()
    plt.cla()
    plt.clf()
    return figure
    '''
    plt.tight_layout() 
    return figure
    '''

def heatmap_from_tensor(data, alphabet, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    figure, ax = plt.subplots(figsize=(8,10))
    #ax = plt.axes()
    px = pd.DataFrame(data, columns=alphabet)
    ff=sns.heatmap(px, linewidth=1, linecolor='w', annot=data, ax = ax)
    ax.set_title(title)
    plt.close()
    plt.cla()
    plt.clf()
    return figure
    #plt.show()