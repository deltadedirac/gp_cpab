import torch
from torch import autograd
from torch import nn
import pdb
import torch.nn.functional as F
#pdb.set_trace()
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
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.MSE = nn.MSELoss()
        self.logGauss = nn.GaussianNLLLoss()

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
        elif method == 'SKL':
            self.loss = ( self.kl_div(input,target) + self.kl_div(target,input) ).sum()
        elif method == 'MSE':
            self.loss = self.MSE(input,target)
        elif method == 'logGauss':
            self.loss = self.logGauss(input,target, torch.ones(*input.shape, requires_grad=True))
        elif method == 'Soft_Label_KLD':
            log_probs = F.log_softmax(input,dim=-1)
            self.component_vals = self.kl_div(log_probs,target)
            self.loss = self.component_vals.mean()
        elif method == 'Soft_Label_KLD_mask':
            masked_idx = (target.argmax(-1).flatten() != self.c2i['-']).nonzero().flatten()
            log_probs = F.log_softmax(input,dim=-1)
            self.loss = self.kl_div(log_probs[:, masked_idx], target[:,masked_idx]).mean()
        elif method == 'JSD':
            #import pdb; pdb.set_trace()
            pa, qa = input, target #input.view(-1, input.size(-1)), target.view(-1, target.size(-1))
            m = (0.5 * (pa + qa))
            self.loss  = 0.5 * (self.kl_div(m,pa) + self.kl_div(m,qa))
            self.component_vals = self.loss
            self.loss = self.loss.mean()
        else:
            self.loss = None

            
        return self.loss

