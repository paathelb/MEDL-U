import torch.nn.functional as F
from torch import nn
import numpy as np
import torch

def get_pred_evidential_aleatoric(out, choose=[0,1,2,3,4,5,6]):
    v, alpha, beta = out
    var = beta / (alpha - 1)

    if len(choose) == 7 or len(choose) == 0:
        pass
    else:
        choose = torch.tensor(choose).long() 
        var = var[:,choose]
    return torch.mean(var, dim=1)

def get_pred_evidential_epistemic(out):
    v, alpha, beta = out
    var = beta / (v * (alpha - 1))
    return torch.mean(var, dim=1)

def get_pred_unc_one_parameter(box_uncertainty, unc = 'aleatoric'):
    if unc == 'aleatoric':
        unc_func = get_pred_evidential_aleatoric
    elif unc == 'epistemic':
        unc_func = get_pred_evidential_epistemic

    unc_loc0 = unc_func(box_uncertainty[0])
    unc_loc1 = unc_func(box_uncertainty[1])
    unc_loc2 = unc_func(box_uncertainty[2])
    unc_dim0 = unc_func(box_uncertainty[3])
    unc_dim1 = unc_func(box_uncertainty[4])
    unc_dim2 = unc_func(box_uncertainty[5])
    unc_rot = unc_func(box_uncertainty[6])
    _unc = (unc_loc0 + unc_loc1 + unc_loc2 + unc_dim0 + unc_dim1 + unc_dim2 + unc_rot)
    return _unc 

def get_pred_evidential_epistemic_one_parameter(box_uncertainty):
    aleatoric_unc_loc0 = get_pred_evidential_epistemic(box_uncertainty[0])
    aleatoric_unc_loc1 = get_pred_evidential_epistemic(box_uncertainty[1])
    aleatoric_unc_loc2 = get_pred_evidential_epistemic(box_uncertainty[2])
    aleatoric_unc_dim0 = get_pred_evidential_epistemic(box_uncertainty[3])
    aleatoric_unc_dim1 = get_pred_evidential_epistemic(box_uncertainty[4])
    aleatoric_unc_dim2 = get_pred_evidential_epistemic(box_uncertainty[5])
    aleatoric_unc_rot = get_pred_evidential_epistemic(box_uncertainty[6])
    aleatoric_unc = (aleatoric_unc_loc0 + aleatoric_unc_loc1 + aleatoric_unc_loc2 + aleatoric_unc_dim0 + aleatoric_unc_dim1 + aleatoric_unc_dim2 + aleatoric_unc_rot)
    return aleatoric_unc 

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self, shift_val=2):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(shift_val)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class LinearNormalGamma(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_chanels, out_channels*4)

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)

    def forward(self, x):
        min_val = 1e-6
        
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, torch.nn.Softplus()(logv) + min_val, torch.nn.Softplus()(logalpha) + min_val + 1, torch.nn.Softplus()(logbeta) + min_val


class LinearNormalGamma_modified(nn.Module):
    def __init__(self, in_chanels, out_channels, act='softplus'):
        super().__init__()
        self.linear = nn.Linear(in_chanels, out_channels*3)
        self.act = act

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)

    def forward(self, x):
        min_val = 1e-6
        
        pred = self.linear(x).view(x.shape[0], -1, 3)
        logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]

        if self.act == 'softplus':
            return torch.nn.Softplus()(logv) + min_val, torch.nn.Softplus()(logalpha) + min_val + 1, torch.nn.Softplus()(logbeta) + min_val
        elif self.act == 'relu':
            return torch.nn.ReLU()(logv) + min_val, torch.nn.ReLU()(logalpha) + min_val + 1, torch.nn.ReLU()(logbeta) + min_val
        elif self.act == 'lrelu':
            return torch.nn.LeakyReLU()(logv) + min_val, torch.nn.LeakyReLU()(logalpha) + min_val + 1, torch.nn.LeakyReLU()(logbeta) + min_val
        elif self.act == 'elu':
            return torch.nn.ELU()(logv) + min_val, torch.nn.ELU()(logalpha) + min_val + 1, torch.nn.ELU()(logbeta) + min_val
        elif self.act == 'silu':
            return torch.nn.SiLU()(logv) + min_val, torch.nn.SiLU()(logalpha) + min_val + 1, torch.nn.SiLU()(logbeta) + min_val
        elif self.act == 'prelu':
            return torch.nn.PReLU()(logv).cuda() + min_val, torch.nn.PReLU()(logalpha).cuda() + min_val + 1, torch.nn.PReLU()(logbeta).cuda() + min_val
 
def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
                - alpha * torch.log(two_blambda) \
                + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
                + torch.lgamma(alpha) \
                - torch.lgamma(alpha + 0.5)
    return nll

def nig_reg(y, gamma, v, alpha, beta, rot=False):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi

def evidential_regression_loss(y, pred, coeff=1.0, weight_loss = None, nll_weight=torch.tensor([1,1,1,3,3,3,1]).float().cuda()):
    gamma, v, alpha, beta = pred

    loss_nll = nig_nll(y, gamma, v, alpha, beta)
    
    if loss_nll.shape[1] == 7:
        loss_nll = loss_nll * nll_weight
    loss_reg = nig_reg(y, gamma, v, alpha, beta)
    
    if weight_loss is None:
        loss_ = loss_nll.mean() + coeff * (loss_reg.mean() - 1e-4)
        return loss_
    else:
        weight_loss = torch.tensor(weight_loss).view(-1,1).cuda()
        loss_ = torch.sum(weight_loss * loss_nll) + coeff * (torch.sum(loss_reg * weight_loss) - 1e-4)
        
        return loss_

class UncertaintyHead(torch.nn.Module):
    def __init__(self, hidden_channels, uncertainty, pred_size, shift_val=2, act='softplus'):
        super(UncertaintyHead, self).__init__()
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus(shift_val)
        
        self.uncertainty = uncertainty
        
        if self.uncertainty == 'False':
            self.lin2 = nn.Linear(hidden_channels // 2, pred_size)
        elif self.uncertainty == 'evidential':
            self.lin2 = LinearNormalGamma(hidden_channels //2, pred_size)
        elif self.uncertainty == 'evidential_LNG_modified':
            self.lin2 = LinearNormalGamma_modified(hidden_channels //2, pred_size, act=act)
        # elif self.uncertainty == 'gaussian':
        #     self.lin2 = LinearNormal(hidden_channels // 2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        if self.uncertainty == 'False':
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            self.lin2.bias.data.fill_(0)

    def forward(self, v, batch=None):
        v = self.lin1(v)
        v = self.act(v)
        
        if self.uncertainty == 'False':
            v = self.lin2(v)
            #u = scatter(v, batch, dim=0)
            u = v
        elif self.uncertainty in ['evidential', 'evidential_LNG_modified', 'gaussian']:
            #u = scatter(v, batch, dim=0)
            u = v
            u = self.lin2(u)
        return u



def modified_mse(gamma, nu, alpha, beta, target, reduction='mean'):
    """
    Lipschitz MSE loss of the "Improving evidential deep learning via multi-task learning."

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        target ([FloatTensor]): true labels.
        reduction (str, optional): . Defaults to 'mean'.

    Returns:
        [FloatTensor]: The loss value. 
    """
    mse = (gamma-target)**2
    c = get_mse_coef(gamma, nu, alpha, beta, target).detach()
    modified_mse = mse*c
    if reduction == 'mean': 
        return modified_mse.mean()
    elif reduction == 'sum':
        return modified_mse.sum()
    else:
        return modified_mse


def get_mse_coef(gamma, nu, alpha, beta, y):
    """
    Return the coefficient of the MSE loss for each prediction.
    By assigning the coefficient to each MSE value, it clips the gradient of the MSE
    based on the threshold values U_nu, U_alpha, which are calculated by check_mse_efficiency_* functions.

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        y ([FloatTensor]): true labels.

    Returns:
        [FloatTensor]: [0.0-1.0], the coefficient of the MSE for each prediction.
    """
    alpha_eff = check_mse_efficiency_alpha(gamma, nu, alpha, beta, y)
    nu_eff = check_mse_efficiency_nu(gamma, nu, alpha, beta, y)
    delta = (gamma - y).abs()
    min_bound = torch.min(nu_eff, alpha_eff).min()
    c = (min_bound.sqrt()/delta).detach()
    return torch.clip(c, min=False, max=1.)


def check_mse_efficiency_alpha(gamma, nu, alpha, beta, y, reduction='mean'):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for alpha, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, alpha).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial alpha(numpy.array) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    
    """
    delta = (y-gamma)**2
    right = (torch.exp((torch.digamma(alpha+0.5)-torch.digamma(alpha))) - 1)*2*beta*(1+nu) / nu

    return (right).detach()


def check_mse_efficiency_nu(gamma, nu, alpha, beta, y):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for nu, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, nu).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial nu(torch.Tensor) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    """
    gamma, nu, alpha, beta = gamma.detach(), nu.detach(), alpha.detach(), beta.detach()
    nu_1 = (nu+1)/nu
    return (beta*nu_1/alpha)