# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/10 14:07:27
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : proc_img.py
# @Description : Some image processing algorithms are implemented, such as image storage and image display.

import matplotlib.pyplot as plt
import numpy as np
def show_img(img,title = None):
    '''
    The shape  of 3D array  supported by the method is (W,H,C) 
    '''
    assert (len(img.shape) == 2 or len(img.shape) == 3), "Image must be 2 or 3 dimensions"
    if len(img.shape) == 3 and img.shape[0] == 3: 
        img =  np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(f"title: {title}")
    plt.axis('off') 
    plt.show()

def save_img(img, title=None, path=None):
    '''
        If img is a 3D array,the its shape must be (W,H,C).
    '''
    assert (len(img.shape) == 2 or len(img.shape) == 3), "Image must be 2 or 3 dimensions"
    if len(img.shape) == 3: 
        if img.shape[0] == 1:
            img =  np.squeeze(img,0)
        elif img.shape[0] == 3:
            img =  np.transpose(img, (1, 2, 0))
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(f"title: {title}")
    plt.axis('off') 
    plt.savefig(path,dpi=600,format='png')
if __name__ == "__main__":
    pass
