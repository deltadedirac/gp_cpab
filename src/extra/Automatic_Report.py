import pdb
import numpy as np
import pandas as pd
#from fpdf import FPDF
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
#pdb.set_trace()
from . import utilities as utilities
#from gp_cpab.src.extra import utilities as utilities
#from gp_cpab import *


def createPDFTemplate():
    pdf = FPDF(orientation = 'P', unit = 'mm', format = 'A4')
    pdf.add_page()
    pdf.set_font('helvetica', 'bold', 10)
    pdf.set_text_color(255, 255, 255)
    return pdf

def tensor2txt(t, path, title):
    t_np = t.detach().numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe

    with open(path, 'a') as file1:
        file1.writelines(title+'\n\n')

    df.to_csv(path, header=None, index=None, sep='\t', mode='a')

    with open(path, 'a') as file1:
        file1.writelines('\n\n\n')



def Generate_Automatic_Report(x1, T, ref_msa, theta_est, theta_estGP, msa_num, msa_numGP, 
                                alphabet, c2i, i2c, i2i, path, config,modeflag, 
                                x1_trans1, x1_trans2, x1_trans3, loss_vals, loss_valsgp,
                                pathX123, indexoutputT, indexlogolinear, indexlogoGP,
                                loss1, loss2, loss3, comp_loss1, comp_loss2, comp_loss3, inverse):

    #pdf = createPDFTemplate()
    #pdb.set_trace()

    with open(pathX123+'/'+indexoutputT, 'a') as file1:
        file1.writelines(config.config.dump() + '\n\n')
        file1.writelines('Alphabet'+'\n')
        file1.writelines(str(alphabet)+'\n\n')
        file1.writelines('Char to int dictionary'+'\n')
        file1.writelines(str(c2i)+'\n\n')
        file1.writelines('Char to int dictionary'+'\n')
        file1.writelines(str(i2c)+'\n\n')

        orig_grid = T.make_grids_for_Regresion(batch_size = x1.shape[0])
        file1.writelines('Original Grid'+'\n')
        file1.writelines(str( orig_grid )+'\n\n')

        file1.writelines('Grid to Transform'+'\n')
        file1.writelines(str( T.grid )+'\n\n')

        file1.writelines('Optimal Theta by using Linear Case - Standard CPAB'+'\n')
        file1.writelines(str( theta_est )+'\n\n')

        file1.writelines('Optimal Theta by using GP Case - GP CPAB'+'\n')
        file1.writelines(str( theta_estGP )+'\n\n')

        if inverse == True:
            theta_est = -theta_est
            theta_estGP = -theta_estGP

        file1.writelines('Grid Transform with theta estimated by LI'+'\n')
        file1.writelines(str( T.transform_grid( orig_grid, theta_est) )+'\n\n')
        
        file1.writelines('Grid Transform with theta estimated by GP'+'\n')
        file1.writelines(str( T.transform_grid( orig_grid, theta_estGP) )+'\n\n')





        file1.writelines('Loss Linear - Linear'+'\n')
        file1.writelines(str( loss1 )+'\n\n')

        file1.writelines('Loss Linear - GP'+'\n')
        file1.writelines(str( loss2 )+'\n\n')

        file1.writelines('Loss GP - GP'+'\n')
        file1.writelines(str( loss3 )+'\n\n')

    tensor2txt(comp_loss1.squeeze(0), pathX123+'/'+indexoutputT, 'loss by component Linear - Linear')
    tensor2txt(comp_loss2.squeeze(0), pathX123+'/'+indexoutputT, 'loss by component Linear - GP')
    tensor2txt(comp_loss3.squeeze(0), pathX123+'/'+indexoutputT, 'loss by component GP - GP')


    tensor2txt(ref_msa.squeeze(0), pathX123+'/'+indexoutputT, 'Reference Sequence in One Hot Encoding')
    tensor2txt(x1.squeeze(0), pathX123+'/'+indexoutputT, 'Raw Data')
    tensor2txt(x1_trans1.detach().squeeze(0), pathX123+'/'+indexoutputT, 'Transformed data using LI and theta optimized with LI')
    tensor2txt(x1_trans2.detach().squeeze(0), pathX123+'/'+indexoutputT, 'Transformed data using GP and theta optimized with LI')
    tensor2txt(x1_trans3.detach().squeeze(0), pathX123+'/'+indexoutputT, 'Transformed data using GP and theta optimized with GP')


    if loss_vals: 
        #utilities.plot_logos_results(x1,ref_msa,x1_trans1, alphabet, c2i, i2c, i2i, msa_num, indexlogolinear,loss_vals, modeflag, pathX123)
        utilities.plot_logos_probs(x1,ref_msa,x1_trans1, alphabet, c2i, i2c, i2i, msa_num, indexlogolinear,loss_vals, modeflag, pathX123)

        
    if loss_valsgp: 
        '''
        utilities.plot_logos_results(x1,ref_msa,x1_trans2, alphabet, c2i, i2c, i2i, msa_numGP, indexlogoGP,loss_valsgp, modeflag, pathX123)
        utilities.plot_logos_results(x1,ref_msa,x1_trans3, alphabet, c2i, i2c, i2i, msa_numGP, indexlogoGP+str(3),loss_valsgp, modeflag, pathX123)
        '''
        utilities.plot_logos_probs(x1,ref_msa,x1_trans3, alphabet, c2i, i2c, i2i, msa_numGP, indexlogoGP+str(3),loss_valsgp, modeflag, pathX123)


    print('done')
