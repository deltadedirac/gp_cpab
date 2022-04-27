import pdb
import numpy as np
import pandas as pd
#from fpdf import FPDF
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utilities import *
#from gp_cpab import *


def createPDFTemplate():
    pdf = FPDF(orientation = 'P', unit = 'mm', format = 'A4')
    pdf.add_page()
    pdf.set_font('helvetica', 'bold', 10)
    pdf.set_text_color(255, 255, 255)
    return pdf

def tensor2txt(t, path, title):
    t_np = t.numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe

    with open(path, 'a') as file1:
        file1.writelines(title+'\n\n')

    df.to_csv(path, header=None, index=None, sep='\t', mode='a')

    with open(path, 'a') as file1:
        file1.writelines('\n\n\n')



def Generate_Automatic_Report(x1, T, ref_msa, theta_est, theta_estGP, msa_num, msa_numGP, 
                                alphabet, c2i, i2c, i2i, path, config,modeflag, 
                                x1_trans1, x1_trans2, x1_trans3, loss_vals, loss_valsgp,
                                pathX123, indexoutputT, indexlogolinear, indexlogoGP):

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

    tensor2txt(ref_msa.squeeze(0), pathX123+'/'+indexoutputT, 'Reference Sequence in One Hot Encoding')
    tensor2txt(x1.squeeze(0), pathX123+'/'+indexoutputT, 'Raw Data')
    tensor2txt(x1_trans1.detach().squeeze(0), pathX123+'/'+indexoutputT, 'Transformed data using LI and theta optimized with LI')
    tensor2txt(x1_trans2.detach().squeeze(0), pathX123+'/'+indexoutputT, 'Transformed data using GP and theta optimized with LI')
    tensor2txt(x1_trans3.detach().squeeze(0), pathX123+'/'+indexoutputT, 'Transformed data using GP and theta optimized with GP')



    if loss_vals: 
        plot_logos_results(x1,ref_msa,x1_trans1, alphabet, c2i, i2c, i2i, msa_num, indexlogolinear,loss_vals, modeflag, pathX123)
        
    if loss_valsgp: 
        plot_logos_results(x1,ref_msa,x1_trans2, alphabet, c2i, i2c, i2i, msa_numGP, indexlogoGP,loss_valsgp, modeflag, pathX123)
        plot_logos_results(x1,ref_msa,x1_trans3, alphabet, c2i, i2c, i2i, msa_numGP, indexlogoGP+str(3),loss_valsgp, modeflag, pathX123)
        #pdb.set_trace()


    print('done')
