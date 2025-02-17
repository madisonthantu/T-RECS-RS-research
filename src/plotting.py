import matplotlib.pyplot as plt
from numpy import reshape
import seaborn as sns
import colorcet as cc
from scipy import interpolate
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_measurements(dfs, parameters_df, ignored_train_ts=0):
    fig, ax = plt.subplots(4, 3, figsize=(15, 15))
    fig.tight_layout(pad=5.0)
    colors = plt.get_cmap('tab10')

    # plot rec_similarity with timesteps on x axis
    legend_lines, legend_names = [], []
    for i, df in enumerate(dfs):
        ts = df['timesteps'][ignored_train_ts:]
        name = parameters_df.loc[i, 'model_name']
        if not np.isnan(parameters_df.loc[i, 'Lambda']):
            name += f" (Lambda: {parameters_df.loc[i, 'Lambda']})"
        legend_names.append(name)
        
        line, = ax[0,0].plot(ts, df['mse'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
        # ax[0,1].plot(ts, df['user_mse'], label=name)
        if 'recall_at_k' in df.columns:
            ax[0,1].plot(ts, df['recall_at_k'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
    
        if 'interaction_spread' in df.columns:
            ax[1,0].plot(ts, df['interaction_spread'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
        if 'inter_cluster_interaction_similarity' in df.columns:
            ax[1,1].plot(ts, df['inter_cluster_interaction_similarity'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
        if 'intra_cluster_interaction_similarity' in df.columns:
            ax[1,2].plot(ts, df['intra_cluster_interaction_similarity'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))

        if 'diversity_metric' in df.columns:
            ax[2,0].plot(ts, df['diversity_metric'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
        if 'inter_cluster_rec_similarity' in df.columns:
            ax[2,1].plot(ts, df['inter_cluster_rec_similarity'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
        if 'intra_cluster_rec_similarity' in df.columns:
            ax[2,2].plot(ts, df['intra_cluster_rec_similarity'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))

        if 'serendipity_metric' in df.columns:
            ax[3,0].plot(ts, df['serendipity_metric'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
        if 'novelty_metric' in df.columns:
            ax[3,1].plot(ts, df['novelty_metric'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
        if 'mean_num_topics' in df.columns:
            ax[3,2].plot(ts, df['mean_num_topics'][ignored_train_ts:], label=name, alpha=0.5, color=colors(i))
        
        legend_lines.append(line)

    for a in ax:
        for b in a:
            b.set_xlabel('Timestep')

    ax[0, 0].set_title('Mean Squared Error')
    ax[0, 0].set_ylabel('MSE')

    ax[0, 1].set_title('Top-5 Recall')
    ax[0, 1].set_ylabel('Recall')
    ax[0, 1].set_xlabel('Timesteps')

    ax[0, 2].set_title('')
    ax[0, 2].set_ylabel('')

    ax[1, 0].set_title('Interaction Spread')
    ax[1, 0].set_ylabel('Jaccard Similarity')

    ax[1, 1].set_title('Inter Cluster Interaction Similarity')
    ax[1, 1].set_ylabel('Jaccard Similarity')

    ax[1, 2].set_title('Intra Cluster Interaction Similarity')
    ax[1, 2].set_ylabel('Jaccard Similarity')

    ax[2, 0].set_title('Diversity')
    ax[2, 0].set_ylabel('Diversity')

    ax[2, 1].set_title('Inter Cluster Recommendation similarity')
    ax[2, 1].set_ylabel('Jaccard Similarity')

    ax[2, 2].set_title('Intra Cluster Recommendation similarity')
    ax[2, 2].set_ylabel('Jaccard Similarity')

    ax[3, 0].set_title('Serendipity')
    ax[3, 0].set_ylabel('Serendipity')

    ax[3, 1].set_title('Novelty')
    ax[3, 1].set_ylabel('Novelty')

    ax[3, 2].set_title('Mean Number of Topics Interacted per User')
    ax[3, 2].set_ylabel('Mean Number of Topics Interacted per User')

    fig.legend(legend_lines,
               legend_names,
               loc='upper center',
               fontsize=12,
               frameon=False,
               ncol=5,
               bbox_to_anchor=(.5, 1.02))


def apply_tsne_2d(x, y, perplexity=50):
    """
    Apply t-SNE to reduce the dimensionality of the data to 2 dimensions.
    Inputs:
        x: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
    Outputs:
        df: pandas dataframe with columns "y", "comp-1", "comp-2"
    """
    tsne = TSNE(perplexity=perplexity,
                n_components=2,
                verbose=0,
                random_state=42)
    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    return df


def plot_clusters(df, axis, palette, previously_seen=[]):
    diff_set = set(df.y.unique()) - set(previously_seen)

    hue_order = list(set(previously_seen)) + list(diff_set)
    
    
    sns.scatterplot(x="comp-1",
                    y="comp-2",
                    hue=df.y.tolist(),
                    hue_order=hue_order,
                    ax=axis,
                    palette=palette,
                    alpha=1,
                    data=df).set(title="")

    # Create hulls
    for i in hue_order:
        points = df[df.y == i][['comp-1', 'comp-2']].values
        if len(points) >= 3:
            # get convex hull
            hull = ConvexHull(points)
            # get x and y coordinates
            # repeat last point to close the polygon
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices,
                                                                0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices,
                                                                1][0])
            # interpolate
            dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 +
                           (y_hull[:-1] - y_hull[1:])**2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull],
                                            u=dist_along,
                                            s=0,
                                            per=1)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            interp_x, interp_y = interpolate.splev(interp_d, spline)
            # plot shape
            axis.fill(interp_x, interp_y, '--', c=palette[i], alpha=0.2)


# def plot_tsne(df, perplexity, n_clusters):
#     """
#     Plots tsne with convex hulls.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Dataframe with columns 'comp-1', 'comp-2' and 'y'
#     perplexity : int
#         Perplexity for tsne
#     """

#     # plot tsne
#     fig, axs = plt.subplots(1, 1, figsize=(15, 5))

#     palette = sns.color_palette(cc.glasbey, n_colors=n_clusters)

#     plot_clusters(df, axs, palette)

#     plt.title(f'TSNE with perplexity={perplexity}')
#     plt.suptitle(f'Opaque points are items. Others are users.')
#     plt.show()

def plot_tsne(df, perplexity, n_clusters, title):
    """
    Plots tsne with convex hulls.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns 'comp-1', 'comp-2' and 'y'
    perplexity : int
        Perplexity for tsne
    """
    # plot tsne
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    palette = sns.color_palette(cc.glasbey, n_colors=n_clusters)
    plot_clusters(df, axs, palette)
    plt.legend(prop={"size": 12}, loc ="upper right")
    return fig


# def plot_tsne_comparison(df1, df2, n_clusters):
#     """
#     Plots two tsne plots (before and after simulation) with convex hulls.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Dataframe with columns 'comp-1', 'comp-2' and 'y'
#     perplexity : int
#         Perplexity for tsne
#     """
#     fig, axs = plt.subplots(1, 2, figsize=(20, 10))

#     palette = sns.color_palette(cc.glasbey, n_colors=n_clusters)

#     plot_clusters(df1, axs[0], palette)
#     axs[0].set_title('Before Simulation')
#     plot_clusters(df2, axs[1], palette, previously_seen=df1.y.unique())
#     axs[1].set_title('After Simulation')

#     plt.suptitle(f'TSNE')
#     plt.show()

# """
# ***************************************
# """
def plot_tsne_comparison(dfs, n_clusters, titles):
    """
    Plots two tsne plots (before and after simulation) with convex hulls.
    Parameters
    ----------
    df : pd.DataFrame
    Dataframe with columns 'comp-1', 'comp-2' and 'y'
    perplexity : int
    Perplexity for tsne
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    palette = sns.color_palette(cc.glasbey, n_colors=n_clusters)
    plot_clusters(dfs[0], axs[0], palette)
    axs[0].set_title(titles[0])
    plot_clusters(dfs[1], axs[1], palette, previously_seen=dfs[1].y.unique())
    axs[1].set_title(titles[1])
    for i, ax in enumerate(axs):
        ax.legend(prop={"size": 13}, loc ="upper left") #fontsize='small')#
        ax.set_title(f"{titles[i]}", fontsize=18)
    axs[0].legend(prop={"size": 13}, loc ="upper left") #fontsize='small')#
    axs[1].legend(prop={"size": 13}, loc ="upper right") #fontsize='small')#        
    return fig

    
    
def plot_item_popularity_distribution(interaction_matrix, y_label="No. interactions", x_label="Item No.", title="Item interaction distribution", ax=None):
    intrxn_per_item = np.sort(np.sum(interaction_matrix, axis=0))[::-1]
    fig = plt.bar(np.arange(intrxn_per_item.size), intrxn_per_item)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return fig


def graph_metrics_by_axis(ax, results, metric_key, metric_key_map):
    ax.plot(results[metric_key])#, label=metric_key)
    ax.set_title(f"{metric_key}")
    ax.set_ylabel(metric_key_map[metric_key])
    ax.set_xlabel("Timestep")
    return ax


def plot_pca_3d(user_representation_df, n_clusters, title="Visualizing Clusters in Three Dimensions Using PCA"):
    pca = PCA(n_components=3)
    PCs_df = pd.DataFrame(pca.fit_transform(user_representation_df.drop(["Cluster"], axis=1)))
    PCs_df.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
    
    plotX = pd.concat([user_representation_df, PCs_df], axis=1, join='inner')
    plotX["dummy"] = 0
    cluster_list = []
    for i in range(n_clusters):
        cluster_list.append(('clust_'+str(i), plotX[plotX["Cluster"] == i]))
    data = []
    for i in range(n_clusters):
        cluster = cluster_list[i]
        data.append(go.Scatter3d(
                        x = cluster[1]["PC1_3d"],
                        y = cluster[1]["PC2_3d"],
                        z = cluster[1]["PC3_3d"],
                        mode = "markers",
                        name = cluster[0],
                        text = None))

    title = title

    layout = dict(title = title,
                xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                )

    fig = dict(data = data, layout = layout)
    return fig


def plot_pca_3d_subplots(user_representation_df, n_clusters, title="Visualizing Clusters in Three Dimensions Using PCA"):
    pca = PCA(n_components=3)
    PCs_df = pd.DataFrame(pca.fit_transform(user_representation_df.drop(["Cluster"], axis=1)))
    PCs_df.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
    
    plotX = pd.concat([user_representation_df, PCs_df], axis=1, join='inner')
    plotX["dummy"] = 0
    cluster_list = []
    for i in range(n_clusters):
        cluster_list.append(('clust_'+str(i), plotX[plotX["Cluster"] == i]))
    data = []
    for i in range(n_clusters):
        cluster = cluster_list[i]
        data.append(go.Scatter3d(
                        x = cluster[1]["PC1_3d"],
                        y = cluster[1]["PC2_3d"],
                        z = cluster[1]["PC3_3d"],
                        mode = "markers",
                        # name = cluster[0],
                        text = None))

    title = title

    layout = go.Layout(title = title,
                xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                )

    fig = go.Figure(data = data, layout = layout)
    return fig