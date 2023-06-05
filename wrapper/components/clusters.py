import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/madisonthantu/Desktop/DREAM/t-recs')
from trecs.base import Component, BaseComponent

import numpy as np

class Clusters():  # pylint: disable=too-many-ancestors
    def __init__(
        self, 
        item_cluster_mapping=None,
        item_cluster_centroids=None,
        user_cluster_mapping=None,
        user_cluster_centroids=None,
        n_clusters=None,
        n_users=None,
        n_items=None,
    ):
        
        if item_cluster_centroids.shape != user_cluster_centroids.shape:
            raise TypeError("Number of item clusters must be equal to number of user clusters")
        
        if item_cluster_mapping==None or item_cluster_mapping.shape[0] != n_items:
            raise TypeError("Number of item cluster mappings must be equal to number of items")
        elif len(np.unique(item_cluster_mapping)[0]) > n_clusters:
            raise TypeError("Cannot have more item cluster mapping values than the number of clusters")
        else:
            self.item_cluster_mapping = item_cluster_mapping
            self.item_cluster_centroids = item_cluster_centroids
        
        if user_cluster_mapping==None or user_cluster_mapping.shape[1] != n_users:
            raise TypeError("Number of user cluster mappings must be equal to number of users")
        elif len(np.unique(user_cluster_mapping)[0]) > n_clusters:
            raise TypeError("Cannot have more user cluster mapping values than the number of clusters")
        else:
            self.user_cluster_mapping = user_cluster_mapping
            self.user_cluster_centroids = user_cluster_centroids
            
        self.n_clusters = n_clusters
        self.n_users = n_users
        self.n_items = n_items