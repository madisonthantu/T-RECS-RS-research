import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../t-recs/')
from trecs.validate import validate_user_item_inputs
from trecs.models import BaseRecommender
from trecs.models import ContentFiltering
from trecs.random import Generator
import trecs.matrix_ops as mo
from wrapper.models.bubble import BubbleBurster
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
from pulp import *



class xQuAD(BubbleBurster):
    """
    Attributes
    -----------
        Inherited from BubbleBurster: :class:`BubbleBurster`
        Inherited from ContentFiltering: :class:`~models.content.ContentFiltering`
        Inherited from BaseRecommender: :class:`~models.recommender.BaseRecommender`
    """
    # We define default values in the signature so we can call the constructor with no argument

    def __init__(self, xquad_method, alpha, **kwargs):
        super().__init__(**kwargs)
        self.xquad_method_str = xquad_method
        if xquad_method == "binary":
            self.xquad_method = self.compute_binary_xquad_scores
        elif xquad_method == "smooth":
            self.xquad_method = self.compute_smooth_xquad_scores
        else:
            raise Exception("Must supply valid method to compute xQuAD scores")
        self.alpha = alpha
        assert(self.probabilistic_recommendations == False), "Cannot use probabalistic_recommendations with the xQuAD model"
        
    def compute_binary_xquad_scores(self, filtered_scores, curr_rec_slate_items, curr_rec_slate_topics, item_topics, user_vec):
        if curr_rec_slate_items is not None:
            accuracy_term = filtered_scores
            diversity_term = curr_rec_slate_topics[user_vec[:,:], item_topics] >= 1
            return (1-self.alpha)*accuracy_term + self.alpha*(1 - diversity_term)
        else:
            xquad_scores = (1-self.alpha) * filtered_scores
            return xquad_scores
    
    def compute_smooth_xquad_scores(self, filtered_scores, curr_rec_slate_items, curr_rec_slate_topics, item_topics, user_vec):
        if curr_rec_slate_items is not None:
            accuracy_term = filtered_scores
            diversity_term = np.divide(curr_rec_slate_topics[user_vec[:,:], item_topics], np.sum(curr_rec_slate_topics, axis=1).reshape((-1,1)))
            return (1-self.alpha)*accuracy_term + self.alpha*(1 - diversity_term)
        else:
            xquad_scores = (1-self.alpha) * filtered_scores
            return xquad_scores
        
    def generate_recommendations(self, k=1, item_indices=None):
        """
       Implementation adapted from: 
            `Managing Popularity Bias in Recommender 
            Systems with Personalized Re-ranking`
       where item popularity (long-tail head v. short head) is applied to the `topics` construct
       studied in this research project.

        Parameters
        -----------

            k : int, default 1
                Number of items to recommend.

            item_indices : :obj:`numpy.ndarray`, optional
                A matrix containing the indices of the items each user has not yet
                interacted with. It is used to ensure that the user is presented
                with items they have not already interacted with. If `None`,
                then the user may be recommended items that they have already
                interacted with.

        Returns
        ---------
            Recommendations: :obj:`numpy.ndarray`
        """
        if item_indices is not None:
            if item_indices.size < self.num_users:
                raise ValueError(
                    "At least one user has interacted with all items!"
                    "To avoid this problem, you may want to allow repeated items."
                )
            if k > item_indices.shape[1]:
                raise ValueError(
                    f"There are not enough items left to recommend {k} items to each user."
                )
        if k == 0:
            return np.array([]).reshape((self.num_users, 0)).astype(int)
        
        s_filtered = mo.to_dense(self.predicted_scores.filter_by_index(item_indices))
        row = np.repeat(self.users.user_vector, item_indices.shape[1])
        row = row.reshape((self.num_users, -1))
        
        curr_rec_slate_topics = np.zeros((self.num_users, self.num_topics))
        item_topics = np.repeat(self.item_topics.reshape((1, -1)), self.num_users, axis=0)
        item_topics = item_topics[row[:, :], item_indices]
        curr_rec_slate = None
        for i in range(self.num_items_per_iter):
            if not np.array_equal(s_filtered < 0, np.isneginf(s_filtered)):
                raise Exception("There are negative predicted scores")
            xquad_scores = self.xquad_method(s_filtered, curr_rec_slate, curr_rec_slate_topics, item_topics, row)
            max_score_item = mo.top_k_indices(xquad_scores, 1, self.random_state)
            max_score_item_id = item_indices[self.users.user_vector, max_score_item.reshape(-1,)].reshape(-1,1)
            # Adding max scoring item to current rec slate
            if curr_rec_slate is not None:
                curr_rec_slate = np.hstack((curr_rec_slate, max_score_item.reshape(-1,1)))
            else:
                curr_rec_slate = max_score_item.reshape(-1,1)
            # Updating topic matrix for current rec slate
            max_item_topics = self.item_topics[max_score_item_id].reshape(-1,)
            curr_rec_slate_topics[self.users.user_vector, max_item_topics] += 1
            np.put_along_axis(s_filtered, max_score_item.reshape((-1,1)), float("-inf"), axis=1)
            
        assert(curr_rec_slate.shape[1] == self.num_items_per_iter), "Size of rec slate does not match num_items_per_iter"
        assert(np.all(np.sum(curr_rec_slate_topics, axis=1) == 10)), "Rec slate topic count does not equal num_items_per_iter"
        
        curr_rec_slate = item_indices[row[:, :k], curr_rec_slate]
        
        if self.is_verbose():
            self.log(f"Item indices:\n{str(item_indices)}")
            self.log(
                f"xQuAD recommendation slate, method={self.xquad_method_str}, alpha={self.alpha}:\n{str(curr_rec_slate)}"
            )
        return curr_rec_slate