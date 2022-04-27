import pdb, torch, math, matplotlib

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch.nn.functional as F
from Bio.SeqIO.FastaIO import SimpleFastaParser
from dataLoaderDiffeo import *
from torch.utils.data import Dataset, DataLoader
from libcpab.libcpab import Cpab

import matplotlib.pyplot as plt
from LossFunctionsAlternatives import LossFunctionsAlternatives
from logomakers import logomaker_plots


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
    trans_data = T.transform_data(raw_data.permute(forw), theta, outsize=outsize )

    rever_data = T.transform_data(trans_data, -theta, outsize=outsize )
    trans_data = trans_data.permute(rev)
    rever_data = rever_data.permute(rev)

    return trans_data, forw, rev

def spatial_transformation2(T, raw_data, ref, theta, modeflag):
    # Transform the images
    forw,rev,outsize = determine_trans_dim2(raw_data,ref,modeflag)
    trans_data = T.transform_data(raw_data.permute(forw), theta, outsize=outsize )

    rever_data = T.transform_data(trans_data, -theta, outsize=outsize )
    trans_data = trans_data.permute(rev)
    rever_data = rever_data.permute(rev)

    return trans_data, forw, rev

'''Main Process'''
if __name__ == "__main__":

    path = 'BLAT1_without_lowers.fasta'
    path_MSA_test = '1_Align_seq_BLAT.aln'
    device = 'cpu'
    modeflag = '1D'
    window_grid = 181#178#174 #200 for 2D

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

    T = Cpab(ndim, backend='pytorch', device=device, zero_boundary=True,
                                             volume_perservation=False, override=False)
    #T.set_solver_params(nstepsolver=150)
    
    x1 = T.backend.to(x1, device=device)
    ref_msa = T.backend.to(ref_msa, device=device)

    theta_ref = T.sample_transformation(ref_msa.shape[0])

    '''
    Experiment one: Alignment of Sequence by using Aligned Observational Sequences as reference'''
    '''-------------------------------------------------------------------------------------------'''

    theta_est = torch.autograd.Variable(T.identity(1, epsilon=1e-6), requires_grad=True)
    #theta_est = T.backend.to(theta_est, device=device)
    #optimizer = torch.optim.Adam([theta_est], lr=0.000001)#, weight_decay=1e-7) #=> FOR CROSSENTROPY
    optimizer = torch.optim.Adam([theta_est], lr=0.01)#, weight_decay=1e-7) #=> FOR MSE
    maxiter = 10000 
    loss_function = LossFunctionsAlternatives()
    pdb.set_trace()
    msa_num = []; loss_vals =[]

    pb = tqdm(desc='Alignment of samples', unit='iters', total=maxiter)
    for i in range(maxiter):
         optimizer.zero_grad()
         #pdb.set_trace()
         #x1_trans, forw_per, rever = spatial_transformation(T, x1, theta_est, modeflag)
         x1_trans, forw_per, rever = spatial_transformation2(T, x1, ref_msa, theta_est, modeflag)
         loss = loss_function(method = 'CE', input = x1_trans, target = ref_msa, forw_per=forw_per)
         loss_vals.append(loss.item())
         loss.backward()
         optimizer.step()
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
            Plotting the logos for RAW, ALIGNED (REF) and results with CPAB
            ---------------------------------------------------------------
            ---------------------------------------------------------------
    '''

    alphabets_logo = [ i if i!='-' else 'X' for i in alphabets]; i2c['0']='X'
    if modeflag == '1D':
      x1_logo_input = x1.argmax(-1).reshape(x1.argmax(-1).shape[1])
      x1_trans_logo_input = x1_trans.argmax(-1).reshape(x1_trans.argmax(-1).shape[1])
      ref_logo_input = ref_msa.argmax(-1).reshape(ref_msa.argmax(-1).shape[1])
    else:
      x1_logo_input = x1.argmax(-1).transpose(2,1).reshape(x1.argmax(-1).transpose(2,1).shape[2])
      x1_trans_logo_input = x1_trans.argmax(-1).transpose(2,1).reshape(x1_trans.argmax(-1).transpose(2,1).shape[2])
      ref_logo_input = ref_msa.argmax(-1).transpose(2,1).reshape(ref_msa.argmax(-1).transpose(2,1).shape[2])


    logos_raw =logomaker_plots( alphabets, c2i, i2c, i2i, torch.stack([ x1_logo_input for i in range(10000)]) ) 
    logos_raw.generate_information_matrix(1)

    logos_alignment =logomaker_plots( alphabets, c2i, i2c, i2i, torch.stack([ ref_logo_input for i in range(10000)]) ) 
    logos_alignment.generate_information_matrix(2)

    #Logo of best one
    best_one = msa_num[loss_vals.index(min(loss_vals))]
    logos_best =logomaker_plots( alphabets, c2i, i2c, i2i, torch.stack([ best_one for i in range(10000)]) ) 
    logos_best.generate_information_matrix(3)

    #logomaker.list_font_names()
    #msa_num = torch.stack(msa_num)
    logos =logomaker_plots( alphabets_logo, c2i, i2c, i2i, torch.stack(msa_num))
    logos.generate_information_matrix(4)
