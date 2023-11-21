# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/10 11:04:01
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : plot.py
# @Description : Some drawing tools are implemented through matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import datasets, manifold
import numpy as np
def plot_line(x, y1,y2,labels = [],colors=[],title="",xlabel="",ylabel="",path=""):
    # 创建直线图
    plt.plot(x, y1, label=labels[0], color=colors[0], linestyle='-', marker='o', markersize=5)
    plt.scatter(x, y1, color='black', marker='o', s=10)  # s参数用于设置圆点的大小
    plt.plot(x, y2, label=labels[1], color=colors[1], linestyle='-', marker='o', markersize=5)
    plt.scatter(x, y2, color='black', marker='o', s=10)  # s参数用于设置圆点的大小
    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()
    if path is not None:
        plt.savefig(path,dpi=600,format='png')
    plt.close()

def plot_2d(points, points_color, title=" ", cmap=None,path=None):
    """
    这里事先不指定坐标，
    
    """
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    x, y = points.T
    col = ax.scatter(x, y, c=points_color, cmap=cmap, s=5, alpha=1.0)
    ax.set_title(title)
    # ax.xaxis.set_major_formatter(ticker.NullFormatter())
    # ax.yaxis.set_major_formatter(ticker.NullFormatter())
    # ax.set_xlim(-100,100)
    # ax.set_ylim(-10,10)
    # plt.xlim(-1, 1)  # 设置 x 轴范围为 [2, 4]
    # plt.ylim(-1, 1)  # 设置 y 轴范围为 [15, 25]
    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()
    if path is not None:
        plt.savefig(path,dpi=600,format='png')
    # 添加颜色条
   
def plot_3d(points, points_color,title=" ", cmap=None, path=None):
    x, y, z = points.T
    fig, ax = plt.subplots(
    figsize=(6, 6),
    facecolor="white",
    tight_layout=True,
    subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, cmap=cmap, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()
    if path is not None:
        plt.savefig(path,dpi=600,format='png')
if __name__ == "__main__":
    """
    test plot_2d
    """
    n_samples = 1500
    circles, y = datasets.make_circles(n_samples)
    S_color = 0.5
    n_components = 2  # number of coordinates for the manifold
    t_sne = manifold.TSNE(
        n_components=n_components,
        perplexity=30,
        init="pca",
        n_iter=250,
        random_state=0,
    )
    S_t_sne = t_sne.fit_transform(circles)
    plot_2d(S_t_sne, S_color, "T-distributed Stochastic  \n Neighbor Embedding")
    
    """
    test plot_3d
    """
    n_samples = 1500
    S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
    n_components = 2  # number of coordinates for the manifold
    t_sne = manifold.TSNE(
        n_components=n_components,
        perplexity=30,
        init="pca",
        n_iter=250,
        random_state=0,
    )
    S_t_sne = t_sne.fit_transform(S_points)
    plot_3d(S_t_sne, S_color, "T-distributed Stochastic  \n Neighbor Embedding")

# Here, a variable parameter x is defined and multiple ndarrray can be passed in to display in the histogram.
def plot_hist(*x, colors = ["blue"], title = "Histogram", xlabel = "", ylabel = ""):
    num_bins = 50
    fig, ax = plt.subplots()
    # the histogram of the data
    if x.dim == 1:
        n, bins, patches = ax.hist(x, num_bins, density=True, color=colors[0])
    elif x.dim == 2:
        for i in range(x.shape[0]):
            n, bins, patches = ax.hist(x[i], num_bins, density=True,color = colors[i])
    n, bins, patches = ax.hist(x, num_bins, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title )
    plt.legend()
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()





