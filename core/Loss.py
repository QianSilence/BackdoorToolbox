import torch
import torch.nn as nn
import torch.nn.functional as F
from models import discriminator

class  EntropyLoss(nn.Module):
    def __init__(self, num_classes=10, reduction="mean"):
        super(EntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
    def forward(self,predict):
        prob_matrix = nn.Softmax()(predict)
        log_likelihood_matrix = nn.LogSoftmax()(predict)
        if self.reduction == "none":
            res = -1.0* torch.sum(log_likelihood_matrix * prob_matrix,dim=1)
        elif self.reduction == "mean":
            res = -1.0* torch.mean(log_likelihood_matrix * prob_matrix)
        # print(f"prob_matrix:{prob_matrix},log_likelihood_matrix:{log_likelihood_matrix},res:{res}\n")
        return res
    
class RKLLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10, prob_min=1e-7, one_hot_min=1e-2, reduction="mean"):
        super(RKLLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.prob_min = prob_min
        self.one_hot_min = one_hot_min
    def forward(self, x, target):
        # one_hot = F.one_hot(target, self.num_classes).float()
        # one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        # ce_loss = -1 * torch.sum(one_hot * torch.log(x), dim=-1)
        rce = RCELoss(num_classes=self.num_classes, prob_min=self.prob_min, one_hot_min=self.one_hot_min, reduction=self.reduction)
        rce_loss = rce(x, target)

        entropy = EntropyLoss(num_classes=self.num_classes, reduction=self.reduction)
        entropy_loss = entropy(x)
        loss = rce_loss - entropy_loss
        # print(f"rce_loss:{rce_loss},entropy_loss:{entropy_loss}\n")
        return loss 
    
class RCELoss(nn.Module):
    """
    Reverse Cross Entropy Loss.
    prob_min=1e-13, one_hot=1e-20
    """
    def __init__(self, num_classes=10, reduction="mean", prob_min=1e-7, one_hot_min=1e-2, device=torch.device("cpu")):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.device = device
        self.prob_min = prob_min
        self.one_hot_min = one_hot_min
    def forward(self, x, target):
        prob = F.softmax(x, dim=-1)
        prob = torch.clamp(prob, min=self.prob_min, max=1.0)
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=self.one_hot_min, max=1.0)
        loss = -1 * torch.sum(prob * torch.log(one_hot), dim=-1)

        if self.reduction == "mean":
            loss = loss.mean()
        # print(f"prob:{prob},one_hot:{one_hot},loss:{loss}\n")
        return loss
    
class SCELoss(nn.Module):
    """Symmetric Cross Entropy."""
    """
    reduction="mean",return tensor(1,)
    reduction="none", return tensor(bitch_size,)
    """

    def __init__(self, alpha=0.1, beta=1.0, num_classes=10, reduction="mean",device=torch.device("cpu")):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        self.device = device
    def forward(self, x, target):
        ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        rce = RCELoss(num_classes=self.num_classes, reduction=self.reduction,device=self.device)
        ce_loss = ce(x, target)
        rce_loss = rce(x, target)
        # bitch_size
        loss = self.alpha * ce_loss + self.beta * rce_loss
        return loss
    
class CBCELoss(nn.Module):
    def __init__(self, beta = 0.0, num_classes=10, n_arr=[], reduction="none",device=torch.device("cpu")):
        super(CBCELoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.n_arr = n_arr 
        self.weights = None
        self.reduction = reduction
        self.device = device

    def update_n_arr(self,n_arr):
        self.n_arr = n_arr
        alphas = []
        self.weights = []
        sum = 0.0
        for n in self.n_arr:
            sum = sum + (1 - self.beta) / (1 - self.beta**n)
            alphas.append((1 - self.beta) / (1-self.beta**n))
        for t in alphas:
            self.weights.append(t * (self.num_classes/sum))
        self.weights = np.array(self.weights) 
        # print(f"sum:{sum},self.num_classes/sum:{self.num_classes/sum}\n") 50
        # print(f"n_arr：{n_arr},sum:{sum},alphas:{alphas},weights:{coef}\n")
        # print(f"indices:{indices.shape},{indices},batch_weights:{batch_weights.shape},{batch_weights}")

    # \frac{1-\beta}{1-\beta^{n_y}}CE(x,target)
    def forward(self, x, target): 
        batch_size = len(x)
        indices = target.cpu().numpy()
        # batch _size
        batch_weights = self.weights[indices] 
        # batch_size  
        ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        ce_loss = ce(x, target) 
        
        # batch_size  * batch _size  ---> 1
        loss =  torch.mean(ce_loss * torch.tensor(batch_weights,device=self.device,requires_grad=False)) 
        return loss
    
class CBSCELoss(nn.Module):
    def __init__(self, beta = 0.0, num_classes=10, n_arr=[], reduction="none",device=torch.device("cpu")):
        super(CBSCELoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.n_arr = n_arr 
        self.weights = None
        self.reduction = reduction
        self.device = device

    def update_n_arr(self,n_arr):
        self.n_arr = n_arr
        alphas = []
        self.weights = []
        sum = 0.0
        for n in self.n_arr:
            sum = sum + (1 - self.beta) / (1 - self.beta**n)
            alphas.append((1 - self.beta) / (1-self.beta**n))
        for t in alphas:
            self.weights.append(t * (self.num_classes/sum))
        self.weights = np.array(self.weights) 
        # print(f"sum:{sum},self.num_classes/sum:{self.num_classes/sum}\n") 50
        # print(f"n_arr：{n_arr},sum:{sum},alphas:{alphas},weights:{coef}\n")
        # print(f"indices:{indices.shape},{indices},batch_weights:{batch_weights.shape},{batch_weights}")

    # \frac{1-\beta}{1-\beta^{n_y}}CE(x,target)
    def forward(self, x, target): 
        batch_size = len(x)
        indices = target.cpu().numpy()
        # batch _size
        batch_weights = self.weights[indices] 
        # batch_size  
        sceloss = SCELoss(alpha=0.1, beta=1.0, reduction="none", device=self.device)
        sce_loss = sceloss(x, target)
        # batch_size  * batch _size  ---> 1
        loss =  torch.mean(sce_loss * torch.tensor(batch_weights,device=self.device,requires_grad=False)) 
        return loss
    
class InfomaxLoss(nn.Module):
    def __init__(self,x_dim = 1024,dim = 10):
        super(InfomaxLoss, self).__init__()
        self.disc = discriminator(z_dim=dim, x_dim=x_dim)
    def forward(self, x_true, z):
        # pass x_true and learned features z from the discriminator
        d_xz = self.disc(x_true, z)
        z_perm = self.permute_dims(z)
        d_x_z = self.disc(x_true, z_perm)
        info_xz = -(d_xz - (torch.exp(d_x_z - 1)))
        return info_xz 
    
    def permute_dims(self, z):
        """
        function to permute z based on indicies
        """
        assert z.dim() == 2
        B, _ = z.size()
        perm = torch.randperm(B)
        perm_z = z[perm]
        return perm_z