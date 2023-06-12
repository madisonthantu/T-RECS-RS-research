import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/madisonthantu/Desktop/DREAM/t-recs')
from trecs.metrics import Measurement
import trecs.matrix_ops as mo
from math import comb

from sklearn.metrics.pairwise import cosine_similarity

# import math
import numpy as np
from itertools import combinations

class NoveltyMetric(Measurement):
    def __init__(self, name="novelty_metric", verbose=False):
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
        """
        The purpose of this metric is to capture the global popularity-based measurements
        - computing the average novelty of all slates that are presented to users at the current timestep
        
        This metric is based on the item diversity measure used in :
        Minmin Chen, Yuyan Wang, Can Xu, Ya Le, Mohit Sharma, Lee Richardson, Su-Lin Wu, and Ed Chi. 
        Values of user exploration in recommender systems. 
        In Proceedings of the 15th ACM Conference on Recommender Systems, 
        RecSys ’21, page 85–95, New York, NY, USA, 2021. Association for Computing Machinery.
        
        Parameters
        ------- -----
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0 or np.sum(recommender.predicted_scores.value) == 0:
            self.observe(None) # no interactions yet
            return
                
        # calculate self information of each item (add eps to avoid log(0) errors)
        item_counts = recommender.item_count
        item_counts[item_counts == 0] = 1
        items_self_info = (-1) * np.log(item_counts)
        
        # turn scores in probability distribution over items to ensure that all independent of the ranking function, the metric yields comparable values
        scores = recommender.predicted_scores.value
        probs = scores / np.sum(scores, axis=1)[:, np.newaxis]     
        
        # get utility of each item given a state of users
        item_states = np.mean(probs, axis=0)
        
        # calculate novelty per item by multiplying self information and utility value
        item_novelties = items_self_info * item_states
        # form sum over all possible items/actions
        item_novelty = np.sum(item_novelties)
        self.observe(item_novelty)
        

class SerendipityMetric(Measurement):
    def __init__(self, name="serendipity_metric", verbose=False):
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
        """
        Metric to capture the unexpectedness/surprise of the recommendation to a specific user.
        Item-wise serendipity is equal to 1 if, for that specific user, that item is associated 
        with a score greater than 0 and the topic cluster of that item is not present in the 
        topic history for that user. Otherwise, item-wise serendipity is equal to 0.
        Global serendipity for an interation is computed as the summation of all item-wise serendipity
        scores divided by the number of users.
        
        This metric is based on the item diversity measure used in :
        Minmin Chen, Yuyan Wang, Can Xu, Ya Le, Mohit Sharma, Lee Richardson, Su-Lin Wu, and Ed Chi. 
        Values of user exploration in recommender systems. 
        In Proceedings of the 15th ACM Conference on Recommender Systems, 
        RecSys ’21, page 85–95, New York, NY, USA, 2021. Association for Computing Machinery.
        
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0:
            self.observe(None) # no interactions yet
            return
        # Boolean matrix where value=1 for the shown items that have a score greater than 0
        user_scores_items_shown = np.take_along_axis(recommender.users.actual_user_scores.value, recommender.items_shown, axis=1) > 0
        # Topics that correspond to each item shown
        topics_shown = recommender.item_topics[recommender.items_shown]
        # Boolean matrix where value=1 if the topic shown is not in the user history, otherwise value=0
        new_topics = np.apply_along_axis(np.isin, 1, topics_shown, recommender.user_topic_history, invert=True)
        # calculate serendipity for all items presented to each user
        items_shown_serendipity = np.multiply(new_topics, user_scores_items_shown)
        # Calculate average serendipity - average serendipity by slate AND users
        self.observe(np.mean(items_shown_serendipity))
       
        
class DiversityMetric(Measurement):
    def __init__(self, name="diversity_metric", verbose=False):
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
        """
        Measures the number of distinct faucets the recommendation set contains, which is measured as 
        the average dissimilarity of all pairs of items in the set is a popular choice.,
        such that: similarity(i, j) = 1 if i and j belongs to the same topic cluster, and 0 otherwise.
        
        This metric is based on the item diversity measure used in :
        Minmin Chen, Yuyan Wang, Can Xu, Ya Le, Mohit Sharma, Lee Richardson, Su-Lin Wu, and Ed Chi. 
        Values of user exploration in recommender systems. 
        In Proceedings of the 15th ACM Conference on Recommender Systems, 
        RecSys ’21, page 85–95, New York, NY, USA, 2021. Association for Computing Machinery.
        
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None) # no interactions yet
            return
        # Getting all possible 2-item combinations (the indices) for the items in a slate
        combos = combinations(np.arange(recommender.num_items_per_iter), 2)
        topic_similarity = np.zeros(recommender.num_users)
        for i in combos:
            # topic_similarity is equal to the number of 2-item combinations in which the items' topics are the same
            item_pair = recommender.items_shown[:, i]
            topic_pair = recommender.item_topics[item_pair]
            topic_similarity += (topic_pair[:,0] == topic_pair[:,1])

        slate_diversity = 1 - ((1 / (recommender.num_items_per_iter * (recommender.num_items_per_iter-1))) * topic_similarity)
        self.observe(np.mean(slate_diversity))


class TopicInteractionMeasurement(Measurement): # TODO: Make this work
    """
    Keeps track of the interactions between users and topics.

    Specifically, at each timestep, it stores a histogram of length
    :math:`|I|`, where element :math:`i` is the number of interactions
    received by topic :math:`i`.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"topic_interaction_histogram"``
            Name of the measurement component.
    """

    def __init__(self, name="topic_interaction_histogram", verbose=False):
        Measurement.__init__(self, name, verbose)


    @staticmethod
    def _generate_interaction_histogram(interactions, num_users, num_topics):
        """
        Generates a histogram of the number of interactions per topics at the
        given timestep.

        Parameters
        -----------
            interactions : :obj:`numpy.ndarray`
                Array of user interactions.

            num_users : int
                Number of users in the system

            num_topics : int
                Number of topics in the system

        Returns
        ---------
            :obj:`numpy.ndarray`:
                Histogram of the number of interactions aggregated by items at the given timestep.
        """
        histogram = np.zeros(num_topics)
        np.add.at(histogram, interactions, 1)
        # Check that there's one interaction per user
        if histogram.sum() != num_users:
            raise ValueError("The sum of interactions must be equal to the number of users")
        return histogram

    def measure(self, recommender):
        """
        Measures and stores a histogram of the number of interactions per
        item at the given timestep.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return

        histogram = self._generate_interaction_histogram(
            recommender.topic_interactions, recommender.num_users, recommender.num_topics
        )
        self.observe(histogram, copy=True)


class MeanNumberOfTopics(Measurement):
    """
    Keeps track of the interactions between users and topics.

    Specifically, at each timestep, it stores a histogram of length
    :math:`|I|`, where element :math:`i` is the number of interactions
    received by topic :math:`i`.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"topic_interaction_histogram"``
            Name of the measurement component.
    """

    def __init__(self, name="mean_num_topics", verbose=False):
        Measurement.__init__(self, name, verbose)


    def measure(self, recommender):
        """
        Measures and stores a histogram of the number of interactions per
        item at the given timestep.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return

        self.observe(np.mean(recommender.user_topic_history.sum(axis=1)))
        
        
class RecallMeasurement(Measurement):
    """
    Measures the proportion of relevant items (i.e., those users interacted with) falling
    within the top k ranked items shown.
    Parameters
    -----------
        k: int
            The rank at which recall should be evaluated.
    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`
        name: str, default ``"recall_at_k"``
            Name of the measurement component.
    """

    # Note: RecallMeasurement evalutes recall for the top-k (i.e., highest predicted value)
    # items regardless of whether these items derive from the recommender or from randomly
    # interleaved items. Currently, this metric will only be correct for
    # cases in which users iteract with one item per timestep

    def __init__(self, k=5, name="recall_at_k", verbose=False):
        self.k = k

        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        Measures the proportion of relevant items (i.e., those users interacted with) falling
        within the top k ranked items shown.
        """
        if self.k >= recommender.num_items_per_iter:
            raise ValueError("k must be smaller than the number of items per iteration")

        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None)  # no interactions yet
            return

        else:
            shown_item_scores = np.take(recommender.predicted_scores.value, recommender.items_shown)
            shown_item_ranks = np.argsort(shown_item_scores, axis=1)
            
            top_k_items = np.empty((len(shown_item_ranks), self.k), dtype=int)
            for i, u in enumerate(recommender.items_shown):
                top_k_items[i] = np.take(u, shown_item_ranks[i, self.k:])
            
            recall = (
                len(np.where(np.isin(recommender.interactions, top_k_items))[0]) / recommender.num_users
            )

        self.observe(recall)

        
class UserMSEMeasurement(Measurement):
    def __init__(self, name="user_mse", verbose=False):
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
        """
        Measures the MSE per user at the current timestep
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        diff = recommender.predicted_scores.value - recommender.users.actual_user_scores.value
        self.observe((diff**2).mean(axis=1))


# class CosineSim(Measurement):
#     def __init__(self, name="cos_sim", verbose=False):
#         Measurement.__init__(self, name, verbose)
        
#     def measure(self, recommender):
#         """
#         Calculate cosine similarity for each user, item pair.
#         """
#         denominator = np.outer(np.linalg.norm(recommender.users_hat.value, axis=1), np.linalg.norm(recommender.items_hat.value, axis=0))
#         # cosine similarity is equal to inner product, divided by the norm of the user & item vector
#         # numerator = mo.inner_product(recommender.users_hat.value, recommender.items_hat.value)
#         numerator = np.dot(recommender.users_hat.value, recommender.items_hat.value)
#         cos_sim = numerator / denominator
#         self.observe(cos_sim)


class AvgIntraClusterCosineSim(Measurement):
    def __init__(self, mapping, n_clusters, name="MeanCosSim_UserCluster", verbose=False):
        self.mapping = mapping
        self.n_clusts = n_clusters
        Measurement.__init__(self, name, verbose)
    
    """
    def measure(self, recommender):
        clusters = np.unique(self.mapping)
        intra_cluster_sim = np.zeros((self.n_clusts, 1))
        for clust in clusters:
            clust_users = np.where(self.mapping == clust)
            clust_users_embed = recommender.users.actual_user_profiles.value[clust_users,:][0]
            denominator = np.outer(np.linalg.norm(clust_users_embed, axis=1), np.linalg.norm(clust_users_embed, axis=1))
            numerator = np.dot(clust_users_embed, clust_users_embed.T)
            cos_sim = numerator / denominator
            if clust_users_embed.shape[0] > 1:
                num_user_pairs = comb(clust_users_embed.shape[0], 2)
                intra_cluster_sim[clust] = np.sum(np.triu(cos_sim)) / num_user_pairs
        self.observe(intra_cluster_sim)
    """
        
    def measure(self, recommender):
    
        for clust in clusters:
            clust_users = np.where(self.mapping == clust)
            clust_users_embed = recommender.users.actual_user_profiles.value[clust_users,:][0]
            denominator = np.outer(np.linalg.norm(clust_users_embed, axis=1), np.linalg.norm(clust_users_embed, axis=1))
            numerator = np.dot(clust_users_embed, clust_users_embed.T)
            cos_sim = numerator / denominator
            if clust_users_embed.shape[0] > 1:
                num_user_pairs = comb(clust_users_embed.shape[0], 2)
                intra_cluster_sim[clust] = np.sum(np.triu(cos_sim)) / num_user_pairs
        self.observe(intra_cluster_sim)


class MeanDistanceFromCentroid(Measurement):
    def __init__(self, user_cluster_ids, user_centroids, name="mean_distance_from_centroid", verbose=False):
        self.user_cluster_ids = user_cluster_ids
        self.user_centroids = user_centroids
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
    
        clusters = np.unique(self.user_cluster_ids)
        avg_clust_dist = np.zeros((self.user_centroids.shape[0], ))
        for clust in clusters:
            clust_users = np.where(self.user_cluster_ids == clust)[0]
            clust_users_embed = recommender.users.actual_user_profiles.value[clust_users,:]
            dist = np.linalg.norm(self.user_centroids[clust] - clust_users_embed, axis=1)
            avg_clust_dist[clust] = np.mean(dist)
        self.observe(avg_clust_dist)
        

class SdWithinCluster(Measurement):
    def __init__(self, user_cluster_ids, user_centroids, name="sd_per_clusters", verbose=False):
        self.user_cluster_ids = user_cluster_ids
        self.user_centroids = user_centroids
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
    
        clusters = np.unique(self.user_cluster_ids)
        sigmas = np.zeros((self.user_centroids.shape[0], 1))
        for clust in clusters:
            clust_users = np.where(self.user_cluster_ids == clust)[0]
            clust_users_embed = recommender.users.actual_user_profiles.value[clust_users,:]
            dist = np.linalg.norm(self.user_centroids[clust] - clust_users_embed, axis=1)
            avg_clust_dist[clust] = np.mean(dist)
        self.observe(avg_clust_dist)