#1.随机初始化的权重来创建这些模型
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
squeezenet = models.squeezenet1_0()
densenet = models.densenet_161()

#2.trochvision还提供了pre-trained的模型，pretrained=True就可以使用预训练的模型
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
                        
#3.由给定URL加载Torch序列化对象。
import torch.utils.model_zoo
state_dict = torch.utils.model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')


from torchvision import datasets
mnist = datasets.mnist






