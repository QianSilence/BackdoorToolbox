# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/10 11:04:01
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : t-SNE.py
# @Description :  Manifold learning in package scikit-learn includes many data dimensionality reduction and visualization algorithms，
# such as T-distributed Stochastic Neighbor Embedding(T-SNE),Principal component analysis Embedding(PCA) etc.Specifically, 
# pelease refer to https://scikit-learn.org/stable/modules/manifold.html
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from plot import plot_2d,plot_3d
from matplotlib import ticker
from sklearn import datasets, manifold
if __name__ == "__main__":
    """
    Comparison of Manifold Learning methods:
    1.Isomap Embedding
    2.Locally Linear Embeddings
    3.Modified Locally Linear Embedding
    4.Hessian Eigenmapping
    5.Spectral Embedding
    6.Local Tangent Space Alignment
    7.Multidimensional scaling (MDS)
    8.T-distributed Stochastic Neighbor Embedding
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
    plot_2d(S_t_sne, S_color, "T-distributed Stochastic  \n Neighbor Embedding")

