"""
t-SNE Reduction for visualization
---------------------------------

A wrapper to SciPy t-SNE algorithm
"""

from sklearn.manifold import TSNE


def tsne_representation(documents_representation, metric, perplexity=30.):
    # a t-SNE model
    # angle value close to 1 means sacrificing accuracy for speed
    # pca initializtion usually leads to better results
    n_iter = 5000
    tsne_model = TSNE(n_components=2, n_iter=n_iter, perplexity=perplexity, verbose=1, random_state=0,
                      init='random', metric=metric)

    # 20-D -> 2-D
    tsne_lda = tsne_model.fit_transform(documents_representation)

    x = tsne_lda[:, 0]
    y = tsne_lda[:, 1]

    return x, y