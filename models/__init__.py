#package:https://docs.python.org/zh-cn/3.7/tutorial/modules.html
#__init__.py 文件将包含该文件的目录转化为包，__init__.py,可以只是一个空文件，但它也可以执行包的初始化代
#码（在加载这个package的时候）
#或设置 __all__或是__path__ 变量
#一个名为 __all__ 的列表，它会被视为在遇到 from package import * 时应该导入的模块名列表。
#__path__ ：指定包中包含的模块和子包的搜索
#dir() 函数查找该包包含的模块

"""
这里将package:attacks package下的模块中的类，变量，函数，之后加载package:attacks中任何这些变量
可以直接加载;print(dir())可以看到 "BadNets"已经当前作用域的有效属性列表之中
"""
from .autoencoder import AutoEncoder
from .baseline_MNIST_network import BaselineMNISTNetwork
from .vgg import *
from .resnet import ResNet


__all__ = ["AutoEncoder","BaselineMNISTNetwork","ResNet"]
__path__=["./","./trainedModels/"]
# print(__all__)
# print(dir())
# print(__path__)

