"""
将package:attacks package下的模块中的类，变量，函数，之后加载package:attacks中任何这些变量
可以直接加载,例如：在test_Badnets.py中;print(dir())可以看到 "BadNets"已经当前作用域的有效属性列表之中
"""
from .Attack import Attack
from .BackdoorAttack import BackdoorAttack
from .BadNets import BadNets
from .RegularBadNets import RegularBadNets
from .AdaptiveBlend import AdaptiveBlend
from .AdaptivePatch import AdaptivePatch
__all__ = ['Attack', 'BackdoorAttack', 'BadNets', 'AdaptiveBlend', 'AdaptivePatch','RegularBadNets']
