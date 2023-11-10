import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils import plot_line
"""
这里分别对比SCE = CE(q|p) + CE(p|q), KL(p(y|x)|q(y|x))

以及对称SKL=KL(q(y|x)|P(y|x)) + KL(p(y|x)|q(y|x))

"""
class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss."""
    def __init__(self, num_classes=10, reduction=None):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-2, max=1.0)
        # print(f"log(one_hot):{torch.log(one_hot)}\n")
        loss = -1 * torch.sum(x * torch.log(one_hot), dim=-1)
        # print(f"(x * torch.log(one_hot):{x * torch.log(one_hot)}\n")
        if self.reduction == "mean":
            loss = loss.mean()
        return loss.item()
    
class SCELoss(nn.Module):
    """Symmetric Cross Entropy."""
    def __init__(self, alpha=0.1, beta=1, num_classes=10, reduction=None):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        """
        x(torch.tensor):input : 包含每个类的得分，2-D tensor,shape为 batch*n
        target(torch.tensor),shape: 大小为 n 的 1—D tensor，包含类别的索引(0到 n-1)。
        """
        # ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        # ce_loss = ce(x,target)

        one_hot = F.one_hot(target, self.num_classes).float()
        # one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        ce_loss = -1 * torch.sum(one_hot * torch.log(x), dim=-1)

        rce = RCELoss(num_classes=self.num_classes, reduction=self.reduction)
        rce_loss = rce(x, target)
        loss = self.alpha * ce_loss + self.beta * rce_loss
        print(f"ce_loss:{ce_loss},rce_loss:{rce_loss}")
        return loss.item()

class  EntropyLoss(nn.Module):
    def __init__(self, num_classes=10, reduction="mean"):
        super(EntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
    def forward(self,x):
        loss = -1.0* torch.sum(x * torch.log(x), dim=1)
        # print(f"entropy:{-1 * x * torch.log(x)}")        
        return loss.item()
    
class RKLLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10, reduction="mean"):
        super(RKLLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
    def forward(self, x, target):
        # one_hot = F.one_hot(target, self.num_classes).float()
        # one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        # ce_loss = -1 * torch.sum(one_hot * torch.log(x), dim=-1)
        rce = RCELoss(num_classes=self.num_classes, reduction=self.reduction)
        rce_loss = rce(x, target)

        entropy = EntropyLoss(num_classes=self.num_classes, reduction=self.reduction)
        entropy_loss = entropy(x)
        loss = rce_loss - entropy_loss
        # print(f"rce_loss:{rce_loss},entropy_loss:{entropy_loss}")
        return loss
    
class SKLLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10, reduction="mean"):
        super(SKLLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
    def forward(self, x, target):
        # ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        # ce_loss = ce(x,target)
        one_hot = F.one_hot(target, self.num_classes).float()
        # one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        ce_loss = -1 * torch.sum(one_hot * torch.log(x), dim=-1)

        kl_loss = ce_loss + 0.0 # 
        rkl = RKLLoss()
        rkl_loss = rkl(x, target)
        loss = self.alpha * ce_loss + self.beta * rkl_loss
        # print(f"ce_loss:{ce_loss},rkl_loss:{rkl_loss},loss:{loss}")
        return loss.item()
    
class CBLoss(nn.Module):
    def __init__(self, beta = 0.0, num_classes=10, reduction="none"):
        super(CBLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
    # \frac{1-\beta}{1-\beta^{n_y}}CE(x,target)
    def forward(self, x, target, class_sample_size):
        
        alphas = []
        coef = []
        sum = 0.0
        for num in class_sample_size:
            sum = sum + (1 - self.beta) / (1 - self.beta**num)
            alphas.append((1-self.beta) / (1-self.beta**num))
        for t in alphas:
            coef.append(t * (self.num_classes/sum)) 
        # print(f"class_sample_size：{class_sample_size},sum:{sum},alphas:{alphas},coef:{coef}\n")
       
        # batch_size  
        ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        ce_loss = ce(x, target)
        
        indices  = target.numpy()
        # batch _size
        batch_coef = np.array(coef)[indices]
        # print(f"indices:{indices.shape},{indices},batch_coef:{batch_coef.shape},{batch_coef}")
        
        # batch_size  * batch _size  ---> 1
        loss = torch.sum(ce_loss * batch_coef)
        return loss
if __name__ == '__main__':
    """
    python test_SCE.py
    CE SCE 以及KL散度的区别
    """
    # alpha = 0.1 
    # beta = 0.0  
    prob = 0.99
    target = 0 # SCE = 0.4710
    # prob = 0.9, target = 0 # SCE = 0.4710, KL = 
    # prob = 0.8, target = 0 # SCE = 0.9433, KL = 
    # prob = 0.7, target = 0 # SCE = 1.4172, KL = 
    # prob = 0.65,target = 0 # SCE =         KL =
    # prob = 0.6  target = 0 # SCE = 1.8931, KL = 
    # prob = 0.55 target = 0 # SCE =         KL = 
    # prob = 0.5  target = 0 # SCE = 2.3718, KL = 
    # prob = 0.4, target = 0 # SCE = 2.8547, KL = 
    # prob = 0.3, target = 0 # SCE = 3.3440, KL = 
           
    # prob = 0.2, target = 0 # SCE = 3.8450,  KL = 
    # prob = 0.1, target = 0 # SCE = 4.3749, KL = 
#...........................................................................................................

    # prob = 0.9, target = 1 # SCE = 4.3785,  KL = 
    # prob = 0.8, target = 1 # SCE = 3.8487,  KL = 
  
    # prob = 0.7, target = 1 # SCE = 3.3477,  KL = 
    # prob = 0.6, target = 1 # SCE = 2.8584,  KL = 
    # prob = 0.55,target = 1 # SCE =          KL = 
    # prob = 0.5, target = 1 # SCE = 2.3755,  KL = 
    # prob = 0.4, target = 1 # SCE = 2.8552,  KL = 

    # prob = 0.3, target = 1 # SCE = 3.3445,  KL = 
    # prob = 0.2, target = 1 # SCE = 3.8455,  KL = 
    # prob = 0.1, target = 1 # SCE = 4.3754,  KL = 
  
    p = prob
    classes = 10
    x = torch.zeros(1,classes)
    x[0][0] = p
    x[0][1:] = (1 - p) / (classes - 1)
    real_target = torch.tensor([target])
    sce = SCELoss(alpha=1.0, beta=1.0)
    sce_loss = sce(x,real_target)
    print(f"loss:{sce_loss}\n")













    # classes = 10
    # real_target = 0
    # fake_target = 1
    # prob_arr = np.arange(0.0,1.0,0.1)
    # real_sce_scores = [0]
    # fake_sce_scores = [0]
    # real_rkl_scores = [0]
    # fake_rkl_scores = [0]
    # real_skl_scores = [0]
    # fake_skl_scores = [0]
    # for i in range(len(prob_arr)):
    #     if i == 0:
    #         continue
    #     p = prob_arr[i]
    #     x = torch.zeros(1,classes)
    #     x[0][0] = p
    #     x[0][1:] = (1 - p) / (classes - 1)

    #     y = torch.zeros(1,classes)
    #     y[0][0] = p
    #     if 1 - p <= p:
    #         y[0][1] = 1 - p  
    #     else:
    #         y[0][1] = p - 1e-4
    #     y[0][2:] = max((1.0 - y[0][0] - y[0][1] ) / (classes - 2),1e-4)

    #     real_target = torch.tensor([real_target])
    #     fake_target = torch.tensor([fake_target])
        
    #     alpha = 0.1
    #     beta = 1.0

    #     sce = SCELoss(alpha=alpha,beta=beta,reduction='sum')
    #     real_SCEscore = sce.forward(x,real_target)
    #     real_sce_scores.append(real_SCEscore)
    #     # print(f"y:{y},fake_target:{fake_target}\n")
    #     fake_SCEscore = sce.forward(y,fake_target)
    #     fake_sce_scores.append(fake_SCEscore)

    #     rkl = RKLLoss(reduction='sum')
    #     real_RKLscore = rkl.forward(x,real_target)
    #     real_rkl_scores.append(real_RKLscore)
    #     fake_RKLscore = rkl.forward(y,fake_target)
    #     fake_rkl_scores.append(fake_RKLscore)

    #     skl = SKLLoss(alpha=alpha,beta=beta,reduction='sum')
    #     real_SKLscore = skl.forward(x,real_target)
    #     real_skl_scores.append(real_SKLscore)
    #     fake_SKLscore = skl.forward(y,fake_target)
    #     fake_skl_scores.append(fake_SKLscore)

    # labels = ["consistent label","inconsistent label"]
    # colors = ["black","red"]
    # xlabel="p"
    # ylabel="sce score"
    
    # dir = "/home/zzq/CreatingSpace/BackdoorToolbox/experiments/ResNet-18_CIFAR-10/Mine_for_BadNets/show/"
    # print(f"prob_arr:{prob_arr.tolist()}\n")
    # title="consistent label sce scores vs inconsistent label sce scores"
    # print(f"real_sce_scores:{real_sce_scores},fake_sce_scores:{fake_sce_scores}\n")
    # path = os.path.join(dir,"sce_scores.png") 
    # plot_line(prob_arr,real_sce_scores,fake_sce_scores,labels=labels,colors=colors,title=title,xlabel=xlabel,ylabel=ylabel,path=path)
   
    # title="consistent label rkl scores vs inconsistent label rkl scores"
    # print(f"real_rkl_scores:{real_rkl_scores},fake_rkl_scores:{fake_rkl_scores}\n")
    # path = os.path.join(dir,"rkl_scores.png") 
    # plot_line(prob_arr,real_rkl_scores,fake_rkl_scores,labels=labels,colors=colors,title=title,xlabel=xlabel,ylabel=ylabel,path=path)

    # title="consistent label skl scores vs inconsistent label skl scores"
    # print(f"real_skl_scores:{real_skl_scores},fake_skl_scores:{fake_skl_scores}\n")
    # path = os.path.join(dir,"skl_scores.png") 
    # plot_line(prob_arr, real_skl_scores,fake_skl_scores,labels=labels,colors=colors,title=title,xlabel=xlabel,ylabel=ylabel,path=path)
    
    # batch_size = 256
    # classes = 10
    # class_sample_size = np.random.randint(50,5000,size=(classes))
    # x = np.random.rand(batch_size,classes)
    # target = np.empty(batch_size)
    # for i in range(10):
    #     indices = np.random.choice(np.arange(batch_size), size = int( batch_size/10), replace=False)
    #     target[indices] = i

    # beta = (1.0-1.0/5000)
    # cbloss = CBLoss(beta=beta, reduction="none")
    # x = torch.tensor(x, dtype=float)
    # target = torch.tensor(target, dtype=int)
    # loss = cbloss(x, target, class_sample_size)
    # print(f"loss:{loss}\n")
 
    

        

