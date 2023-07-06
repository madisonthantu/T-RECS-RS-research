import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/madisonthantu/Desktop/DREAM/t-recs')
from trecs.metrics import Measurement, Diagnostics
import trecs.matrix_ops as mo

from sklearn.metrics.pairwise import cosine_similarity

# import math
import numpy as np
from itertools import combinations


class MeanCosineSim(Measurement, Diagnostics):
    def __init__(self, pairs, name="mean_cosine_sim", verbose=False, diagnostics=False):
        self.diagnostics = diagnostics
        self.pairs = pairs
        self.first_user = [p[0] for p in pairs]
        self.second_user = [p[1] for p in pairs]
        Measurement.__init__(self, name, verbose)
        if diagnostics:
            Diagnostics.__init__(self)

    def measure(self, recommender):
        cos_sim_matrix = cosine_similarity(recommender.users.actual_user_profiles.value)
        paired_cos_sim = cos_sim_matrix[self.first_user, self.second_user]
        self.observe(np.mean(paired_cos_sim))
        if self.diagnostics:
            self.diagnose(paired_cos_sim)


class MeanDistanceFromCentroid(Measurement, Diagnostics):
    def __init__(self, user_cluster_ids, user_centroids, name="mean_distance_from_centroid", verbose=False, diagnostics=False):
        self.diagnostics = diagnostics
        self.user_ids = np.arange(user_cluster_ids.shape[0])
        self.cluster_ids = user_cluster_ids
        self.user_centroids = user_centroids
        Measurement.__init__(self, name, verbose)
        if diagnostics:
            Diagnostics.__init__(self)
        
    def measure(self, recommender):
        if self.user_centroids.shape[0] == 1:
            centroid_vecs = np.repeat(self.user_centroids, recommender.users.actual_user_profiles.value.shape[0], axis=0)
        else:
            centroid_vecs = self.user_centroids[self.cluster_ids]
        dist = np.linalg.norm(recommender.users.actual_user_profiles.value - centroid_vecs, axis=1)
        self.observe(np.mean(dist))
        if self.diagnostics:
            self.diagnose(dist)
        

class MeanCosineSimPerCluster(Measurement, Diagnostics):
    def __init__(self, user_cluster_ids, n_clusts, name="mean_cosine_sim_per_cluster", verbose=False, diagnostics=False):
        self.diagnostics = diagnostics
        self.user_cluster_ids = user_cluster_ids
        self.n_clusts = n_clusts
        assert (np.unique(user_cluster_ids).shape[0] == n_clusts), "User cluster assignment does not match number of clusters"
        Measurement.__init__(self, name, verbose)
        if diagnostics:
            Diagnostics.__init__(self)

    def measure(self, recommender):
        avg_cos_sim_per_clust = list()
        
        for clust in range(self.n_clusts):
            clust_users = np.where(self.user_cluster_ids == clust)[0]
            user_embeddings = recommender.users.actual_user_profiles.value[clust_users, :]
            cos_sim_matrix = cosine_similarity(user_embeddings)
            cos_sim_idxs = np.triu_indices(cos_sim_matrix.shape[0], k=1)
            cos_sim_vals = cos_sim_matrix[cos_sim_idxs[0], cos_sim_idxs[1]]
            avg_cos_sim_per_clust.append(np.mean(cos_sim_vals))
            
        self.observe(avg_cos_sim_per_clust)
        if self.diagnostics:
            self.diagnose(np.array(avg_cos_sim_per_clust))
    
    
class MeanDistanceFromCentroidPerCluster(Measurement, Diagnostics):
    def __init__(self, user_cluster_ids, user_centroids, n_clusts, name="mean_distance_from_centroid_per_cluster", verbose=False, diagnostics=False):
        self.diagnostics = diagnostics
        self.user_cluster_ids = user_cluster_ids
        self.cluster_ids = user_cluster_ids
        self.user_centroids = user_centroids
        self.n_clusts = n_clusts
        Measurement.__init__(self, name, verbose)
        if diagnostics:
            Diagnostics.__init__(self)
        
    def measure(self, recommender):
        avg_dist_per_clust = list()
        for clust in range(self.n_clusts):
            clust_users = np.where(self.user_cluster_ids == clust)[0]
            user_embeddings = recommender.users.actual_user_profiles.value[clust_users, :]
            dist = np.linalg.norm(np.subtract(user_embeddings, self.user_centroids[clust]), axis=1)
            avg_dist_per_clust.append(np.mean(dist))
        
        self.observe(avg_dist_per_clust)
        if self.diagnostics:
            self.diagnose(np.array(avg_dist_per_clust))