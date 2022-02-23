import numpy as np
import pandas as pd
import pdb
import torch
import matplotlib.pyplot as plt

from logomakers import logomaker_plots








def plot_GP_components(x1_trans,T,channel,name, scalefactor=10):

    with torch.no_grad():
        #x=torch.linspace(0,x1_trans[0,:,0].shape[0]-1,x1_trans[0,:,0].shape[0])

        for i in range(channel):

            f, ax = plt.subplots(1, 1, figsize=(9, 8))
            # Get upper and lower confidence bounds
            lower, upper = T.lower, T.upper
            # Plot predictive means as blue line
            #ax.set_yscale('log')
            #ax.set_xscale('log')
            x = T.cpabgrid.numpy()*scalefactor
            pdb.set_trace()
            ax.plot(x, x1_trans[0,:,i].numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(x, lower[:,i].numpy(), upper[:,i].numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.set_xlim([0, 10])
            ax.legend([ 'Mean'])
            plt.xticks(x)
            plt.yticks(np.arange(-2.5, 2.5, step=0.2))
            plt.title('GP_component_'+str(i))
            plt.savefig(name+str(i)+'.png')
            plt.show()


def plot_GP_components3(x1_trans,T,channel,name, scalefactor=10, mode = 'see_train'):

    with torch.no_grad():
        #x=torch.linspace(0,x1_trans[0,:,0].shape[0]-1,x1_trans[0,:,0].shape[0])

        for i in range(channel):

            f, ax = plt.subplots(1, 1, figsize=(9, 8))
            # Get upper and lower confidence bounds
            #lower, upper = T.lower, T.upper
            
            if mode == 'see_train':
                x = T.cpabgrid
                x_smoothness = torch.linspace(0,torch.max(x),100)
                y_pred = T.predict_operation(x_smoothness)
                y_pred_mean = y_pred.mean.unsqueeze(0)
                lower, upper = y_pred.confidence_region()
                ax.plot(x_smoothness, y_pred_mean[0,:,i].numpy(), 'b')
                ax.fill_between(x_smoothness, lower[:,i].numpy(),  upper[:,i].numpy(), alpha=0.3)    
            else:
                lower, upper = T.lower,T.upper
                x = T.ugrid.numpy()
                ax.plot(x, x1_trans[0,:,i].numpy(), 'b')
                ax.fill_between(x, lower[:,i].numpy(),  upper[:,i].numpy(), alpha=0.3)


            pdb.set_trace()
            ax.scatter(x, x1_trans[0,:,i].numpy(), s=30, label="Samples", zorder=20) # Shade between the lower and upper confidence bounds
            #ax.fill_between(x, lower[:,i].numpy(),  upper[:,i].numpy(), alpha=0.3)
            ax.set_ylim([-4, 4])
            #ax.set_xlim([0, 10])
            ax.legend([ 'Mean'])
            plt.xticks(x)
            #plt.yticks(np.arange(-2.5, 2.5, step=0.2))
            plt.title('GP_component_'+str(i) )
            plt.savefig(name+str(i)+'.png')
            plt.show()




def plot_GP_components2(x1_trans,T,channel,name):
    with torch.no_grad():
        x=torch.linspace(0,x1_trans[0,:,0].shape[0]-1,x1_trans[0,:,0].shape[0])
        x2=torch.linspace(0,x1_trans[0,:,0].shape[0]-1,10)


    with gpytorch.settings.fast_pred_var():
        x1_trans2 = T.likelihood_Multitask(T.multiout_GP_Interpolator(x2)).mean

    with torch.no_grad():
        for i in range(channel):

            f, ax = plt.subplots(1, 1, figsize=(4, 3))
            # Get upper and lower confidence bounds
            lower, upper = T.lower, T.upper
            # Plot predictive means as blue line

            ax.plot(x.numpy(), x1_trans[0,:,i].numpy(), 'b')
            ax.plot(x2.numpy(), x1_trans2[:,i].numpy(), 'c')
            # Shade between the lower and upper confidence bounds
            #ax.fill_between(x.numpy(), lower[:,i].numpy(), upper[:,i].numpy(), alpha=0.5)
            ax.set_ylim([-0.1, 0.5])
            ax.legend([ 'Mean','GP continuous'])
            plt.xticks(x)
            plt.savefig('data/'+name+str(i)+'.png')
            plt.show()


def plot_logos_results(x1,ref_msa,x1_trans, alphabets, c2i,i2c,i2i,msa_num, path, loss_vals, modeflag):    
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
    logos_raw.generate_information_matrix('original_'+path)

    logos_alignment =logomaker_plots( alphabets, c2i, i2c, i2i, torch.stack([ ref_logo_input for i in range(10000)]) ) 
    logos_alignment.generate_information_matrix('ref_'+path)

    #Logo of best one
    best_one = msa_num[loss_vals.index(min(loss_vals))]
    #logos_best =logomaker_plots( alphabets, c2i, i2c, i2i, torch.stack([ best_one for i in range(10000)]) ) 
    logos_best =logomaker_plots( alphabets, c2i, i2c, i2i, torch.stack([ x1_trans_logo_input for i in range(10000)]) ) 
    logos_best.generate_information_matrix('best_'+path)

    #logomaker.list_font_names()
    #msa_num = torch.stack(msa_num)
    logos =logomaker_plots( alphabets_logo, c2i, i2c, i2i, torch.stack(msa_num))
    logos.generate_information_matrix('dist_'+path)


def create_deformation_movie(ugrid, grids, paths, fitnesval, iter = 0):
    f, ax = plt.subplots(1, 1, figsize=(9, 8))
    x = grids[0,0]
    y = np.array([0.0 for i in range(len(x))])

    #pdb.set_trace()

    ax.set_ylim(-0.5,0.5)
    ax.scatter(ugrid, y+0.1)
    ax.plot(ugrid, y+0.1, 'c')


    ax.scatter(grids[0,0], y)
    ax.plot(x, y, 'c')

    ax.set_title('Grid Shifting CPAB, iter = ' + str(iter)+', CE = '+ str(fitnesval))
    ax.legend(['Plane', 'Grid Points'])
    plt.savefig(paths)
    plt.close()
    plt.cla()
    plt.clf()
