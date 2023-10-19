# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/10 11:04:01
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : plot.py
# @Description : Some drawing tools are implemented through matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import datasets, manifold

def plot_2d(points, points_color, title=" ", cmap=None,path=None):
    """
    这里事先不指定坐标，
    
    """
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    x, y = points.T
    col = ax.scatter(x, y, c=points_color, cmap=cmap, s=30, alpha=1.0)
    ax.set_title(title)
    # ax.xaxis.set_major_formatter(ticker.NullFormatter())
    # ax.yaxis.set_major_formatter(ticker.NullFormatter())
    # ax.set_xlim(-100,100)
    # ax.set_ylim(-100,100)
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



