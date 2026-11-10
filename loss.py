
#from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import torch
import torch.nn.functional as F
def get_adaptive_alpha(current_step,total_step,type="static",sigmoid_range=6,decay_k=5):
    if type == "decrease":
        alpha = (1-current_step/total_step)  ##[1,0]
    elif type == "decrease_exp":
        t=current_step/total_step
        alpha = np.exp(-decay_k*t)
    elif type == "decrease_cosine":
        t=current_step/total_step
        alpha = (1 + math.cos(math.pi * t)) / 2
    elif type == "decrease_sigmoid":
        value = (current_step/total_step) 
        value = 2*sigmoid_range*value-sigmoid_range  ## [-3,3]
        alpha =  1/(1+math.exp(value))
    elif type == "increase":
        alpha =  (current_step/total_step)
    elif type == "static":
        pass
    else:
        raise NotImplementedError()
    return alpha
def get_adaptive_beta(current_value,current_step,total_step,type="step",beta_max=0.2,beta_min=0,
                      sigmoid_range=6):
    change = beta_max - beta_min
    if type == "decrease":
        beta = beta_min+change* (1-current_step/total_step)

    elif type == "decrease_sigmoid":
        value = current_step/total_step #[0,1]
        value = 2*sigmoid_range*value-sigmoid_range  ## [-3,3]
        beta =  beta_min+ change* 1/(1+math.exp(value))

    elif type == "static":
        beta = current_value
    else:
        raise NotImplementedError()
    return beta
def policy_entropy(logits):
    """
    Compute the entropy of the policy given logits.
    
    Args:
        logits (torch.Tensor): Output logits from the network (before softmax).
                               Shape: (batch_size, num_classes).
    
    Returns:
        entropy (torch.Tensor): Entropy of the policy for each input in the batch.
                                Shape: (batch_size,).
    """
    # Compute softmax probabilities
    #probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits,dim=-1)
    probs = torch.exp(log_probs)
    
    # Compute entropy: -sum(p * log(p))
    #log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    

    return entropy
class CELoss(nn.Module):
    def __init__(self,  temperature=1,beta=0):
        super(CELoss, self).__init__()
        self.temperature = temperature
        self.beta=beta
        self.alpha = 1
        

    def forward(self, logits, targets):
        pre_entropy = policy_entropy(logits)
        logits = logits / self.temperature
        post_entropy = policy_entropy(logits)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        #loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        loss =     ce_loss
        loss += -1.0*self.beta*post_entropy

        return loss.mean(),loss,pre_entropy.mean(),post_entropy.mean()



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, label_smooth_alpha=0.1, num_classes=None, temperature=1):
        """
        Args:
            alpha: Smoothing factor (0 = no smoothing, 1 = uniform distribution).
            num_classes: Number of classes (required if targets are not one-hot).
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = label_smooth_alpha
        self.num_classes = num_classes
        self.temperature = temperature

    def forward(self, logits, targets):
        pre_entropy = policy_entropy(logits)
        logits = logits / self.temperature
        post_entropy = policy_entropy(logits)
        num_classes = logits.size(-1)

        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=num_classes).float()

        # Apply label smoothing
        smoothed_targets = (1 - self.alpha) * targets + self.alpha / num_classes

        # Compute cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smoothed_targets * log_probs).sum(dim=-1)

        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # else:
        #     return loss

        return loss.mean(),loss,pre_entropy.mean(),post_entropy.mean()
class Symmetric_KL_Loss(nn.Module):
    def __init__(self,  A=8,temperature=1):
        super(Symmetric_KL_Loss, self).__init__()
        self.alpha = 1
        self.temperature = temperature
        self.A=A
        

    def forward(self, logits, targets):
        pre_entropy = policy_entropy(logits)
        logits = logits / self.temperature
        post_entropy = policy_entropy(logits)



        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        self.dim = -1
        log_prob = logits.log_softmax(dim=self.dim)
        pt = torch.exp(log_prob)
        pt_nograd=pt.detach()
        pt_target=pt_nograd[range(len(targets)),targets]
        rl_loss =   pt_target*ce_loss
        loss = 0.5 * ce_loss + 0.5 *(self.A*rl_loss - post_entropy)
        return loss.mean(),loss,pre_entropy.mean(),post_entropy.mean()
class PG_CELoss(nn.Module):
    def __init__(self,  PG_alpha,temperature=1):
        super(PG_CELoss, self).__init__()
        self.alpha = PG_alpha
        self.temperature = temperature
        self.beta=0
        

    def forward(self, logits, targets):
        pre_entropy = policy_entropy(logits)
        logits = logits / self.temperature
        post_entropy = policy_entropy(logits)



        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        self.dim = -1
        log_prob = logits.log_softmax(dim=self.dim)
        pt = torch.exp(log_prob)
        pt_nograd=pt.detach()
        pt_target=pt_nograd[range(len(targets)),targets]
        rl_loss =   pt_target*ce_loss
        loss = self.alpha*ce_loss + (1-self.alpha)*rl_loss

        return loss.mean(),loss,pre_entropy.mean(),post_entropy.mean()

class FocalLoss(nn.Module):
    def __init__(self,  gamma=2,temperature=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.beta=0
        self.eps=1e-7
        

    def forward(self, logits, targets):
        pre_entropy = policy_entropy(logits)
        logits = logits / self.temperature
        post_entropy = policy_entropy(logits)
        #ce_loss = F.cross_entropy(logits, targets, reduction='none')
        log_prob = logits.log_softmax(dim=-1)
        log_prob_y = log_prob[range(len(targets)),targets]
        pt = torch.exp(log_prob_y)
        pt = torch.clamp(pt, self.eps, 1 - self.eps)
        #loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        loss = -(1-pt)**self.gamma * log_prob_y
        #loss = (  (1 - pt) ** self.gamma * ce_loss)

        return loss.mean(),loss,pre_entropy.mean(),post_entropy.mean()

    