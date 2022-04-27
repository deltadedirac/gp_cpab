import torch
from torch import autograd
from torch import nn
import pdb
'''
    The inheritance to nn.Module is made with the purpose of obtaining 
    flexibility when it will be necessary to do a custom Loss Function
'''

class LossFunctionsAlternatives(nn.Module):
    """
    - This criterion choose a specifical loss function for optimizing the diffeomorphical transformation.
    
    - The inheritance of nn.Module template is just for creating customized loss functions in case 
      of being necessary.
    """

    def __init__(self):
        super().__init__()
        self.Cross_Entropy = nn.CrossEntropyLoss(reduction = 'sum')
        self.L1Loss = nn.L1Loss()
        self.kl_div = nn.KLDivLoss()
        self.MSE = nn.MSELoss()

    def get_dictionaries_to_mask_data(self, c2i, i2c, i2i):
        self.c2i = c2i
        self.i2c = i2c
        self.i2i = i2i

    def forward(self, method, input, target, forw_per):
        #pdb.set_trace()
        if method == 'CE':
            self.loss= self.Cross_Entropy(input.permute(forw_per), target.argmax(-1))
        elif method == 'CEmask':
            #print('from here')
            #pdb.set_trace()# just take all the indexes excepting the ones that contains the gap value
            masked_idx = (target.argmax(-1).flatten() != self.c2i['-']).nonzero().flatten()
            self.loss= self.Cross_Entropy(input[: , masked_idx].permute(forw_per), target[: , masked_idx].argmax(-1))
            #print('done')

        elif method == 'L1':
            self.loss = self.L1Loss(input,target)
        elif method == 'KL':
            self.loss = self.kl_div(input,target)
        elif method == 'MSE':
            self.loss = self.MSE(input,target)
        else:
            self.loss = None

            
        return self.loss

