import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pickle
import os
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from importlib import reload
import itertools
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(1, '/Users/madisonthantu/Desktop/DREAM/t-recs')
from trecs.metrics import MSEMeasurement, InteractionSpread, InteractionSpread, InteractionSimilarity, RecSimilarity, RMSEMeasurement, InteractionMeasurement
from trecs.components import Users
import trecs.matrix_ops as mo

from chaney_utils import *

sys.path.insert(1, '../')
from wrapper.models.bubble import BubbleBurster
from wrapper.metrics.evaluation_metrics import *
from wrapper.metrics.clustering_metrics import MeanCosineSim, MeanDistanceFromCentroid, MeanCosineSimPerCluster, MeanDistanceFromCentroidPerCluster
from src.utils import load_and_process_movielens, compute_embeddings, user_topic_mapping, create_cluster_user_pairs, compute_constrained_clusters

random_state = np.random.seed(42)

import argparse
import errno
import warnings
import pprint
import pickle as pkl
warnings.simplefilter("ignore")


def run_bubble_burster(user_representation, item_representation, item_cluster_ids, user_cluster_ids, user_cluster_centers, global_user_cluster_ids, global_user_cluster_centers, drift, attention_exp, retrain, global_user_pairs, inter_cluster_user_pairs, intra_cluster_user_pairs, args, rng):
    
    users = Users(
        actual_user_profiles=user_representation, 
        repeat_interactions=False,
        drift=drift,
        attention_exp=attention_exp
    )
    measurements = [
        MSEMeasurement(),  
        InteractionSpread(),                
        InteractionSimilarity(pairs=global_user_pairs, name='global_interaction_similarity'), 
        InteractionSimilarity(pairs=inter_cluster_user_pairs, name='inter_cluster_interaction_similarity'), 
        InteractionSimilarity(pairs=intra_cluster_user_pairs, name='intra_cluster_interaction_similarity'), 
        
        MeanCosineSim(pairs=global_user_pairs, name='mean_global_cosine_sim'),
        MeanCosineSim(pairs=intra_cluster_user_pairs, name='mean_intra_cluster_cosine_sim'),
        MeanCosineSim(pairs=inter_cluster_user_pairs, name='mean_inter_cluster_cosine_sim'),
        MeanCosineSimPerCluster(user_cluster_ids=user_cluster_ids, n_clusts=args["num_clusters"], name="mean_cosine_sim_per_cluster"), 
        
        MeanDistanceFromCentroid(user_cluster_ids=user_cluster_ids, user_centroids=user_cluster_centers, name="mean_cluster_distance_from_centroid"), 
        MeanDistanceFromCentroid(user_cluster_ids=global_user_cluster_ids, user_centroids=global_user_cluster_centers, name="mean_global_distance_from_centroid"), 
        MeanDistanceFromCentroidPerCluster(user_cluster_ids=user_cluster_ids, user_centroids=user_cluster_centers, n_clusts=args["num_clusters"], name="mean_distance_from_centroid_per_cluster")
    ]
    bubble = BubbleBurster(
        actual_user_representation=users, 
        actual_item_representation=item_representation,
        item_topics=item_cluster_ids,
        num_attributes=args["num_attrs"],
        num_items_per_iter=10,
        # seed=rng,
        record_base_state=True
    )
    bubble.add_metrics(*measurements)
    bubble.startup_and_train(timesteps=args["startup_iters"])
    bubble.run(timesteps=args["sim_iters"], train_between_steps=args["repeated_training"])
    bubble.close() # end logging
    return bubble

    
"""
python run_param_exp.py --output_dir param_exp_results/single_training  --startup_iters 10 --sim_iters 50 --num_sims 3 --single_training
python run_param_exp.py --output_dir param_exp_results/repeated_training  --startup_iters 10 --sim_iters 50 --num_sims 3 --repeated_training

python run_param_exp.py --output_dir param_exp_results/50train50run  --startup_iters 50 --sim_iters 50 --num_sims 3
python run_param_exp.py --output_dir param_exp_results/10train90run  --startup_iters 10 --sim_iters 90 --num_sims 3

python run_param_exp.py --output_dir param_exp_results/with_clustering_metrics/50train50run  --startup_iters 50 --sim_iters 50 --num_sims 3
python run_param_exp.py --output_dir param_exp_results/with_clustering_metrics/10train90run  --startup_iters 10 --sim_iters 90 --num_sims 3
"""
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='running parameter experiments')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=random_state)
    parser.add_argument('--num_attrs', type=int, default=20)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--num_sims', type=int, default=5)
    parser.add_argument('--startup_iters', type=int, required=True)
    parser.add_argument('--sim_iters', type=int, required=True)
    parser.add_argument('--repeated_training', dest='repeated_training', action='store_true')
    parser.add_argument('--single_training', dest='repeated_training', action='store_false')
    parser.add_argument('--attention_exp', type=float, default=-0.8)
    parser.add_argument('--max_iter', type=int, default=1000)

    parsed_args = parser.parse_args()
    args = vars(parsed_args)

    print("Creating experiment output folder... 💻")
    # create output folder
    try:
        os.makedirs(args["output_dir"])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # write experiment arguments to file
    with open(os.path.join(args["output_dir"], "args.txt"), "w") as log_file:
        pprint.pprint(args, log_file)

    rng = np.random.default_rng(args["seed"])
    
    hyper_params = {"drift":[0.0, 0.05, 0.1], "attention_exp":[0, -0.8], "repeated_training":[0,1]}
    models = dict([(f"{p[0]}drift_{p[1]}attention_{p[2]}retraining", p) for p in itertools.product(*hyper_params.values())])
    
    metric_list = ["mse", "interaction_spread", "global_interaction_similarity", "inter_cluster_interaction_similarity", "intra_cluster_interaction_similarity"]
    result_metrics = {k: defaultdict(list) for k in metric_list}
    
    binary_ratings_matrix = load_and_process_movielens(file_path='/Users/madisonthantu/Desktop/DREAM/data/ml-100k/u.data')

    print("Running simulations...👟")
    # for i in tqdm(range(args["num_sims"])):
    for i in range(args["num_sims"]):
        print(f"SIMULATION {i}")
        # Get user and item representations using NMF
        user_representation, item_representation = compute_embeddings(binary_ratings_matrix, n_attrs=args["num_attrs"], max_iter=args["max_iter"])
        # Define topic clusters using NMF
        item_cluster_ids, item_cluster_centers = compute_constrained_clusters(item_representation.T, name='item_clusters', n_clusters=args["num_clusters"], n_attrs=args["num_attrs"], max_iter=args["max_iter"])
        user_cluster_ids, user_cluster_centers = compute_constrained_clusters(user_representation, name='user_clusters', n_clusters=args["num_clusters"], n_attrs=args["num_attrs"], max_iter=args["max_iter"])
        global_user_cluster_ids, global_user_cluster_centers = compute_constrained_clusters(user_representation, name='global_user_clusters', n_clusters=1)
        # Get user pairs - global user pairs, intra-cluster user pairs, inter-cluster user pairs
        global_user_pairs = []
        num_users = len(user_cluster_ids)
        for u_idx in range(num_users):
            global_user_pairs += [(u_idx, v_idx) for v_idx in range(u_idx+1, num_users)]
        inter_cluster_user_pairs, intra_cluster_user_pairs = create_cluster_user_pairs(user_cluster_ids)
        
        for params in models:
            print(params)
            drift, attention_exp, retrain = models[params]
            model = run_bubble_burster(
                user_representation=user_representation, 
                item_representation=item_representation, 
                item_cluster_ids=item_cluster_ids, 
                user_cluster_ids=user_cluster_ids,
                user_cluster_centers=user_cluster_centers,
                global_user_cluster_ids=global_user_cluster_ids, 
                global_user_cluster_centers=global_user_cluster_centers,
                drift=drift,
                attention_exp=attention_exp,
                retrain=retrain,
                global_user_pairs=global_user_pairs, 
                inter_cluster_user_pairs=inter_cluster_user_pairs, 
                intra_cluster_user_pairs=intra_cluster_user_pairs,
                args=args,
                rng=rng
            )
            for metric in metric_list:
                result_metrics[metric][params].append(process_measurement(model, metric))
        
    # write results to pickle file
    output_file = os.path.join(args["output_dir"], "sim_results.pkl")
    pkl.dump(result_metrics, open(output_file, "wb"), -1)
    print("Done with simulation! 🎉")