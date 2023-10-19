# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/13 17:35:23
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_Mine.py
# @Description  : Backdoor Defense via Mutual Information Neural Estimation 
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from torchvision import transforms 
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# print(sys.path)
from core.attacks import BadNets,BackdoorAttack
from core.base import Observer, Base, MineTrainer
from models import BaselineMNISTNetwork
import random
import time
import datetime
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import manifold
from utils import show_img,save_img,accuracy,compute_accuracy,get_latent_rep, plot_2d
from utils import Log,parser,count_model_predict_digits
""".
需要进一步验证的实验：

1.展示现有防御方案的潜在分离效果，是否出现后门样本的真实回归？后门效果有没有得到抑制？

（1）

生成潜在表示并可视化

2.后门样本中主特征和后门特征是否存在tradeoff关系？softmax得到的p(y|x)是否有所体现？
需要对比展示一下正常样本和后门样本的p(y|x)方差的差异，主要是概率分布的差异，
如熵信息


3.在监督学习下，潜在表示各分类的聚类分布是如何变化的？方差是如何变化的？均值是如何变化的？

需要展示一下方差和均值的变化曲线

"""
# ========== Set global settings ==========
global_seed = 333
deterministic = False
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = BASE_DIR + '/datasets/'
date = datetime.date.today()
work_dir = os.path.join(BASE_DIR,'experiments/BaselineMNISTNetwork_MNIST_BadNets_MINE')
datasets_dir = os.path.join(work_dir,'datasets')
poison_datasets_dir = os.path.join(datasets_dir,'poisoned_MNIST')
latents_dir = os.path.join(datasets_dir,'latents')
predict_dir = os.path.join(datasets_dir,'predict')
model_dir = os.path.join(work_dir,'model')
show_dir = os.path.join(work_dir,'show')
dirs = [work_dir, datasets_dir, poison_datasets_dir, latents_dir, predict_dir, model_dir, show_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
# ========== BaselineMNISTNetwork_MNIST_Mine ==========
# The basic data type in torch is "tensor". In order to be computed, other data type, like PIL Image or numpy.ndarray,
#  must be Converted to "tensor".
dataset = torchvision.datasets.MNIST
transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
classes =  trainset.classes
transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
optimizer = torch.optim.SGD
pattern = torch.zeros((28, 28), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((28, 28), dtype=torch.float32)
weight[-3:, -3:] = 1.0
schedule = {
    'experiment': 'BaselineMNISTNetwork_MNIST_BadNets_MINE',
    "train_strategy":MineTrainer,
    # Settings for reproducible/repeatable experiments
    'seed': global_seed,
    'deterministic': deterministic,
    # Settings related to device
    'device': "cpu",
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    # Settings related to tarining 
    'pretrain': None,
    'epochs': 30,
    # 256，512
    'batch_size': 128,
    'num_workers': 2,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],
    # Settings aving model,data and logs
    'work_dir': work_dir,
    'log_iteration_interval': 100,
    # 日志的保存路径 work_dir+experiment+当前时间
}
"""
自定义 loss函数

得输入神经网络的输出，潜在变量Z，真实标签

中间的一切过程由loss函数内部实现

输出loss值

问题来了，不同的loss含数据计算，需要不同的输入值怎么办呢，那么如何做呢？

可以自定义个对象前向传播对象，在训练中调用其前向传播函数即可

"""

class Loss(nn.Module):
    # 超参数初始化，如
    def __init__(self, alpha, beta, regularize = False, regularize_layer = None):
        super(Loss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.softmax = nn.Softmax()
        self.cross_entropy_Loss = nn.CrossEntropyLoss()
        self.regularize = regularize,
        self.regularize_layer = regularize_layer
        # print(self.regularize_layer)
    # 一般是预测值和label
    """
    Notice:
    *
    torch.mul(a, b)
    torch.mm(a, b)
    torch.matmul(a, b)
    

    n:the number of samples
    k: k classes

    gamma_matrix
    torch.Size([128, 10])
    pi_vector
    torch.Size([10])
    mu_matrix
    torch.Size([10, 10])
    middle_matrix
    torch.Size([10, 128, 10])
    sigma_matrix
    torch.Size([10, 10, 10])

    """
    def entropy(self,input):
        prob_matrix = nn.Softmax()(input)
        log_likelihood_matrix = nn.LogSoftmax()(input)
        entropy_vector = -1.0* torch.sum(log_likelihood_matrix * prob_matrix,dim=1)
        return entropy_vector
        
    def forward(self, latents_z, predict, label):
        # n*k ---> n*k
        gamma_matrix= self.softmax(predict)
        # n*k ---> k
        N_vector = torch.sum(gamma_matrix, 0) 
        N = gamma_matrix.size()[0]
        pi_vector  = N_vector / N
        
        # k*n * (n*m) ---> k*m
        mu_matrix = torch.mm(torch.transpose(gamma_matrix,0,1), latents_z) 

        # k*n*m - k*1*m  ---->  k*n*m 
        delta_matrix = latents_z.unsqueeze(0) - mu_matrix.unsqueeze(1)

        # gamma_matrix: n*k--> k*n*1 ---> k*n*1 *  k*n*m ---> k*n*m
        middle_matrix = torch.transpose(gamma_matrix,0,1).unsqueeze(2) * delta_matrix
        #  k*m*n * k*n*m ---> k*m*m
        sigma_matrix = torch.matmul(delta_matrix.permute(0, 2, 1),middle_matrix) 

        # m*m
        diag_matrix = 0.5 * torch.eye(sigma_matrix.size()[1],sigma_matrix.size()[2])
      

        # k*m*m + 1*m*m ---> k*m*m
        sigma_matrix = sigma_matrix + diag_matrix.unsqueeze(0)
        # (k*m)^ * k*1 ---> m*1
        overline_mu = torch.mm(torch.transpose(mu_matrix,0,1),pi_vector.unsqueeze(1)) 
      
        #  k*m  -  1*m ---> k*m
        delta_matrix2 = mu_matrix -  torch.transpose(overline_mu,0,1)

        # k*m  * k*1 ---> k*m
        middle_matrix2 = torch.mul(delta_matrix2, pi_vector.unsqueeze(1))
        
        #  (k*m)^T * k*m ---> m*m
        sigma_matrix2 = torch.mm(delta_matrix2.permute(0, 1),middle_matrix2) 
       
        # m*m
        diag_matrix2 = 0.5 * torch.eye(sigma_matrix2.size()[0],sigma_matrix2.size()[1])
        sigma_matrix2 = sigma_matrix2 + diag_matrix2

        # torch.no_grad():
            # if not torch.cholesky(sigma_matrix)
        # Determinant = torch.det(sigma_matrix)
        # print("Determinant:{}".format(Determinant))
        # print("pi_vector:{}, shape: {},sum:{},".format(pi_vector,pi_vector.shape,torch.sum(pi_vector)))
        # indexs = torch.le(Determinant,0)

        # key = np.where(np.array(indexs) == True)
        # if torch.any(indexs).item():
            # print(indexs)
            # print(Determinant[indexs])
            # print("mu:{}".format(torch.norm(mu_matrix,1)))
            # print("det:{}".format(Determinant))
            # print("pi_vector:{}".format(pi_vector))

            # max_pi, index = torch.max(pi_vector,dim=0)

            # print("index:{0},max_pi_index:{1},max_pi:{2}".format(key,index,max_pi))
            # print("pi:{}".format(pi_vector[indexs]))
            
               
            # # print(sigma_matrix[indexs])
            # print(sigma_matrix.shape)
            # print(sigma_matrix[indexs].shape)
            # matrix = sigma_matrix[indexs].squeeze(0)

            # # 判断是否对称
            # symmetry = (torch.transpose(matrix,0,1) - matrix) < 0.1
            # print("Is matrix symmetry:{0}".format(symmetry))
            # # 判断是否正定
            # eigenvalues, _ = torch.eig(matrix)
            # is_positive_definite = all(eigenvalues[:, 0] > 0)
            # if not is_positive_definite:
            #     print(is_positive_definite)
            #     print("eigenvalues:{},Determinant:{},mu:{}".format(eigenvalues,Determinant[indexs],torch.norm(mu_matrix,1)))

            #     print("gamma_matrix:{},shape:{}".format(gamma_matrix,gamma_matrix.shape))
            #     print("pi_vector:{},shape: {}".format(pi_vector,pi_vector.shape))
            # exit(-1)

        loss1 = self.cross_entropy_Loss(predict,label)


        # k*1 * k*1 --> 1
        loss2 = torch.sum(pi_vector * torch.log(torch.det(sigma_matrix)))
        # loss3 = torch.abs(60 - torch.log(torch.det(sigma_matrix2)))
        loss3 = 0.0 
        # loss = loss1 + self.beta * loss2
        # return (loss,loss1,loss2)
        
        # print("det:{0}".format(torch.det(sigma_matrix2)))
        # print("log det:{0}".format(torch.log(torch.det(sigma_matrix2))))
        # print("loss3:{0}".format(loss3))
        entropy_vector = self.entropy(predict)
        # print(entropy_vector.shape)
        # print(entropy_vector)
        vector = torch.max(torch.tensor(0), 0.5 - entropy_vector)
        loss4 = torch.sum(vector*vector).item()

        loss = loss1 + self.beta * (loss2 + loss3) + self.alpha * loss4
        # print("loss:{0},loss1:{1},loss2:{2},loss3:{3},loss4:{4}".format(loss,loss1,loss2,loss3,loss4))
        return (loss,loss1,loss2,loss3,loss4,entropy_vector)
    

task = {
    'train_dataset': trainset,
    'test_dataset' : testset,
    'model' : BaselineMNISTNetwork(),
    'optimizer': optimizer,
    'loss' : Loss(0.1, 0.01, regularize = True, regularize_layer="fc2"),
}

# # Parameters are needed according to attack strategy
attack_schedule ={ 
    'experiment': 'BaselineMNISTNetwork_MNIST_BadNets_MINE',
    'attack_strategy': 'BadNets',
    # attack config
    'y_target': 1,
    'poisoning_rate': 0.05,
    'pattern': pattern,
    'weight': weight,
    'poisoned_transform_index': 0,
    'poisoned_target_transform_index': 0,
    # device config
    'device': None,
    'CUDA_VISIBLE_DEVICES': None,
    'GPU_num': None,
    'batch_size': None,
    'num_workers': None,
    # Settings related to saving model,data and logs
    'work_dir': work_dir,
    'train_schedule':schedule,
}

if __name__ == "__main__":
    """
    Users can select the task to execute by passing parameters.
    1. The task of generating and showing backdoor samples
        python test_Mine.py --task "generate backdoor samples"

    2. The task of showing backdoor samples
        python test_Mine.py --task "show train backdoor samples"
        python test_Mine.py --task "show test backdoor samples"
        
    3. The task of training backdoor model
        python test_Mine.py --task "attack"

    4.The task of testing backdoor model
        python test_Mine.py --task "test"

    5.The task of generating latents
        python test_Mine.py --task "generate latents"

    6.The task of visualizing latents by t-sne
        python test_Mine.py --task "visualize latents by t-sne"
        python test_Mine.py --task "visualize latents for target class by t-sne"
    7.The task of comparing predict_digits
        python test_Mine.py --task "generate predict_digits"
        python test_Mine.py --task "compare predict_digits"

    """
    log = Log(osp.join(work_dir, 'log.txt'))
    experiment = attack_schedule['experiment']
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)

    badnets = BadNets(
        task,
        attack_schedule
    )
    backdoor = BackdoorAttack(badnets)
    # Show the structure of the model
    print(task['model'])
    args = parser.parse_args()
    if args.task == "generate backdoor samples":
        # Generate backdoor sample
        log("\n==========Generate backdoor samples==========\n")
        poisoned_train_dataset = backdoor.get_poisoned_train_dataset()
        poisoned_test_dataset = backdoor.get_poisoned_test_dataset()
        poison_indices = poisoned_train_dataset.get_poison_indices()
        benign_indexs = list(set(range(len(poisoned_train_dataset))) - set(poison_indices))
        log("Total samples:{0}, poisoning samples:{1}, benign samples:{2}".format(len(poisoned_train_dataset),\
        len(poison_indices),len(benign_indexs)))
        #Save poisoned dataset
        torch.save(poisoned_train_dataset, os.path.join(poison_datasets_dir,'train.pt'))
        torch.save(poisoned_test_dataset, os.path.join(poison_datasets_dir,'test.pt'))

    elif args.task == "show train backdoor samples":
        log("\n==========Show posioning train sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        poison_indices = poisoned_train_dataset.get_poison_indices()
        print(len(poison_indices))
        index = poison_indices[random.choice(range(len(poison_indices)))]
        image, label = poisoned_train_dataset[index] 
        # Outside of neural networks, packages including numpy and matplotlib are usually used for data operations, 
        # so the type of data is usually converted to np.ndrray()
        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_train_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)

    elif args.task == "show test backdoor samples":
        log("\n==========Show posioning test sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poison_indices = poisoned_test_dataset.get_poison_indices()
        index = poison_indices[random.choice(range(len(poison_indices)))]
        # print(len(poison_indices))
        image, label = poisoned_test_dataset[index] 
        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_test_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)

    elif args.task == "attack":
        #Train and get backdoor model
        log("\n==========Train on poisoned_train_dataset and get backdoor model==========\n")
        poisoned_model = backdoor.get_backdoor_model()
        torch.save(poisoned_model.state_dict(), os.path.join(model_dir, 'backdoor_model.pth'))
        log("Save backdoor model to" + os.path.join(model_dir, 'backdoor_model.pth'))
    
    elif args.task == "test":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of backdoor attack on poisoned_test_dataset==========\n")
        # poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        testset = poisoned_test_dataset
        poisoned_test_indexs = list(testset.get_poison_indices())
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
        #Alreadly exsiting trained model
        model = BaselineMNISTNetwork()
        print(os.path.join(model_dir, 'backdoor_model.pth'))
        model.load_state_dict(torch.load(os.path.join(model_dir, 'backdoor_model.pth')),strict=False)
        predict_digits, labels = backdoor.test(model=model, test_dataset=testset)

        benign_accuracy = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
        poisoning_accuracy = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(testset),len(poisoned_test_indexs),\
                                                                                len(benign_test_indexs)))                                                                                                                                                
        log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_accuracy,poisoning_accuracy))
    
    elif args.task == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        #Alreadly exsiting trained model and poisoned datasets
        # device = torch.device("cuda:1")
        # model = BaselineMNISTNetwork()
        device = torch.device("cpu")
        model = BaselineMNISTNetwork()
        # model.to(device)
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)
        # Get the latent representation in the middle layer of the model.
        layer = "fc2"
        latents,y_labels = get_latent_rep(model, layer, poisoned_train_dataset, device=device)
        latents_path = os.path.join(latents_dir,"MNIST_train_latents.npz")
        print(type(latents))
        print(latents.shape)
        np.savez(latents_path, latents=latents, y_labels=y_labels)
        # n*k ---> 
    elif args.task == "visualize latents by t-sne":
        log("\n========== Clusters of latent representations for all classes.==========\n")  
        # # Alreadly exsiting latent representation
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        latents_path = os.path.join(latents_dir,"MNIST_train_latents.npz")
        data = np.load(latents_path)
        latents,y_labels = data["latents"],data["y_labels"]
   
        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=30, early_exaggeration=120, init="pca", n_iter=250, random_state=0 )
        # print(latents.shape)
        points = t_sne.fit_transform(latents)
        # print(points.shape)
        # points = points*1000
        # print(t_sne.kl_divergence_)
        #Display data clusters for all category by scatter plots
        num = len(classes)
        # Custom color mapping
        colors = [plt.cm.tab10(i) for i in range(num)]
        
        # print(colors)
        colors.append("black")  
        y_labels[poison_indices] = num
        # Create a ListedColormap objectall_
        cmap = mcolors.ListedColormap(colors)
        title = "t-SNE diagram of latent representation"
        path = os.path.join(show_dir,"latent_2d_all_clusters.png")
        plot_2d(points, y_labels, title=title, cmap=cmap, path=path)
           
    elif args.task == "visualize latents for target class by t-sne":
        #Clusters of latent representations for target class
        log("\n==========Verify the assumption of latent separability.==========\n")  
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()

        latents_path = os.path.join(latents_dir,"MNIST_train_latents.npz")
        # Alreadly exsiting latent representation
        data = np.load(latents_path)
        latents,y_labels = data["latents"],data["y_labels"]

        indexs = np.where(y_labels == attack_schedule["y_target"])[0]
        # print(len(indexs))
        # print(len(poison_indices))
        color_numebers = [0 if index in poison_indices else 1 for index in indexs]
        color_numebers = np.array(color_numebers)

        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=30, early_exaggeration=120, init="pca", n_iter=250, random_state=0 )
        points = t_sne.fit_transform(latents[indexs])
        points = points*100

        colors = ["blue","red"]
        cmap = mcolors.ListedColormap(colors)
        title = "t-SNE diagram of latent representation"
        path = os.path.join(show_dir,"latent_2d_target_cluster.png")
        plot_2d(points, color_numebers, title=title, cmap=cmap, path=path)
    elif args.task == "generate predict_digits":
        log("\n==========generate predict_digits.==========\n") 
        device = torch.device("cpu")
        model = BaselineMNISTNetwork()
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        model.load_state_dict(torch.load(os.path.join(model_dir, 'backdoor_model.pth')),strict=False)
        testset = poisoned_train_dataset
        predict_digits, labels = backdoor.test(model=model, test_dataset=testset)
        predict_digits_path = os.path.join(predict_dir,"MNIST_test_predict_digits.npz")
        # print(type(predict_digits))
        # print(predict_digits.shape)
        np.savez(predict_digits_path, predict_digits=predict_digits.numpy(), y_labels=labels.numpy())
        
    elif args.task == "compare predict_digits": 
        log("\n==========compare predict_digits.==========\n") 
       
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        predict_digits_path = os.path.join(predict_dir,"MNIST_test_predict_digits.npz")
        data = np.load(predict_digits_path)
        predict_digits,y_labels = data["predict_digits"],data["y_labels"]

        save_path = os.path.join(predict_dir,"MNIST_test_predict_digits_statistics.npz")
        y_target = attack_schedule["y_target"]
        res = count_model_predict_digits(predict_digits,y_labels,poison_indices,y_target,save_path)
        y_target = res["y_target"]
        predicts, y_post_prob_matrix, entropy_matrix, entropy_vector = res[y_target]["predicts"], res[y_target]["y_post_prob_matrix"],\
            res[y_target]["entropy_matrix"],res[y_target]["entropy_vector"]
        poison_predicts, poison_y_post_prob_matrix, poison_entropy_matrix, poison_entropy_vector = res[y_target]["poison_predicts"],\
            res[y_target]["poison_y_post_prob_matrix"], res[y_target]["poison_entropy_matrix"], res[y_target]["poison_entropy_vector"]
        index = random.choice(range(len(predicts)))
        poison_index = random.choice(range(len(poison_predicts)))
        log("y_target:{0},index:{1},poison_index:{2}\n".format(y_target,index,poison_index)) 

        log("\n==========predicts vs poison_predicts.==========\n") 
        log("predict:{0}\n".format(predicts[index]))
        log("poison predict:{0}\n".format(poison_predicts[poison_index])) 

        log("\n==========y_post_prob_matrix vs poison_y_post_prob_matrix.==========\n") 
        log("p(y|x):{0}\n".format(y_post_prob_matrix[index]))
        log("poison p(y|x):{0}\n".format(poison_y_post_prob_matrix[poison_index])) 
        
        log("\n==========entropy vs poison_entropy.==========\n") 
        log("entropy:{0},sum:{1}\n".format(entropy_matrix[index],entropy_vector[index]))
        log("poison entropy:{0},sum:{1}\n".format(poison_entropy_matrix[poison_index],poison_entropy_vector[poison_index])) 

   
   
       

       


       




       
    
    
 

 
    
  


    
  

 

  
  

