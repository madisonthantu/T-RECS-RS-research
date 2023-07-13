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
from trecs.metrics import MSEMeasurement, InteractionSpread, InteractionSpread, InteractionSimilarity, RMSEMeasurement, InteractionMeasurement
from trecs.components import Users
import trecs.matrix_ops as mo

from representation_experiments.surprise_utils import compute_embeddings_surprise

sys.path.insert(1, '/Users/madisonthantu/Desktop/DREAM/T-RECS-RS-research')
from prelim_experiments.param_experiments.chaney_utils import *
from wrapper.models.bubble import BubbleBurster
from wrapper.metrics.evaluation_metrics import DiversityMetric, NoveltyMetric, TopicInteractionMeasurement, TopicInteractionSpread, UserMSEMeasurement
from wrapper.metrics.clustering_metrics import MeanCosineSim, MeanDistanceFromCentroid, MeanCosineSimPerCluster, MeanDistanceFromCentroidPerCluster
from src.utils import compute_constrained_clusters, create_global_user_pairs, user_topic_mapping, create_cluster_user_pairs, load_and_process_movielens, compute_embeddings
from src.post_processing_utils import process_diagnostic

random_state = np.random.seed(42)

import argparse
import errno
import warnings
import pprint
import pickle as pkl
warnings.simplefilter("ignore")


def run_bubble_burster(
    user_representation, 
    item_representation,
    item_cluster_ids,
    metrics_list,
    param_dict,
    args
    ):
    
    users = Users(
        actual_user_profiles=user_representation, 
        repeat_interactions=False,
        drift=param_dict["drift"],
        attention_exp=param_dict["attention_exp"]
    )
    bubble = BubbleBurster(
        actual_user_representation=users, 
        actual_item_representation=item_representation,
        item_topics=item_cluster_ids,
        num_attributes=param_dict["num_attrs"],
        num_items_per_iter=10,
        record_base_state=True
    )
    bubble.add_metrics(*metrics_list)
    bubble.startup_and_train(timesteps=args["startup_iters"])
    bubble.run(timesteps=args["sim_iters"], train_between_steps=param_dict["repeated_training"])
    bubble.close() # end logging
    return bubble

    
"""
python run_prelim_exp.py --output_dir prelim_exp_results/repeated_training  --startup_iters 10 --sim_iters 90 --num_sims 1
python run_prelim_exp.py --output_dir prelim_exp_results/repeated_training_user_cluster_mapping  --startup_iters 10 --sim_iters 90 --num_sims 1
python run_prelim_exp.py --output_dir prelim_exp_results/repeated_training_user_cluster_mapping_10clusters  --startup_iters 10 --sim_iters 90 --num_sims 1

python run_prelim_exp.py --output_dir prelim_exp_results/single_training  --startup_iters 50 --sim_iters 50 --num_sims 1
"""
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='running parameter experiments')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=random_state)
    parser.add_argument('--num_sims', type=int, default=5)
    parser.add_argument('--startup_iters', type=int, required=True)
    parser.add_argument('--sim_iters', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--max_iter', type=int, default=1000)

    parsed_args = parser.parse_args()
    args = vars(parsed_args)
    
    if os.path.isdir(args["output_dir"]):
        print("The supplied output directory already exists. Do you wish to overwrite this directory's contents? [y/n]: ")
        if str(input()).lower() != "y":
            sys.exit()
        
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
    
    hyper_params = {
        "drift":[0.1], 
        "attention_exp":[-0.8], 
        "repeated_training":[1], 
        "num_attrs":[20], 
        "num_clusters":[10],
        "compute_embeddings_via_surprise":[0],
        "create_cluster_user_pairs_by_user_topic_mapping":[0]
    }

    models = dict()
    for p in itertools.product(*hyper_params.values()):
        models[f"{p[0]}drift_{p[1]}attention_{p[2]}retraining_{p[3]}attributes_{p[4]}clusters_{p[5]}surprise_{p[6]}user-topic-mapping"] = dict(zip(hyper_params.keys(), p))
        
    metric_names = [
        "mse",
        "interaction_spread",
        "global_interaction_similarity",
        "inter_cluster_interaction_similarity",
        "intra_cluster_interaction_similarity",
        "mean_global_cosine_sim",
        "mean_intra_cluster_cosine_sim",
        "mean_inter_cluster_cosine_sim",
        "mean_cosine_sim_per_cluster",
        "mean_cluster_distance_from_centroid",
        "mean_global_distance_from_centroid",
        "mean_distance_from_centroid_per_cluster",
        "interaction_histogram",
        "rmse",
        "mean_novelty",
        "mean_slate_topic_diversity",
        "topic_interaction_histogram",
        "topic_interaction_spread",
        "mse_per_user"
    ]
    result_metrics = {k: defaultdict(list) for k in metric_names}
    
    diagnostic_metrics = set((
        "mse",
        "global_interaction_similarity",
        "inter_cluster_interaction_similarity",
        "intra_cluster_interaction_similarity",
        "mean_global_cosine_sim",
        "mean_intra_cluster_cosine_sim",
        "mean_inter_cluster_cosine_sim",
        "mean_cosine_sim_per_cluster",
        "mean_cluster_distance_from_centroid",
        "mean_global_distance_from_centroid",
        "mean_novelty"
    ))
    diagnostics_vars = ["mean", "std", "median", "min", "max", "skew"]
    model_diagnostics = {k: defaultdict(list) for k in diagnostic_metrics}
    result_diagnostics = {k: defaultdict(list) for k in diagnostic_metrics}
    
    sim_environment_variables = [
        "actual_user_representation_initial",
        "actual_user_representation_final",
        "user_cluster_assignments",
        "user_cluster_centroids",
        "item_representation",
        "item_cluster_assignments",
        "item_cluster_centroids",
        "global_user_centroid",
        "user_item_cluster_mapping"
    ]
    sim_environment = {k: defaultdict(list) for k in sim_environment_variables}

    print("Running simulations...👟")
    # for i in tqdm(range(args["num_sims"])):
    for i in range(args["num_sims"]):
        
        
        for count, param_string in enumerate(models, start=1):
            print(f"Model [{count}/{len(models)}]: {param_string}")
            param_vals = models[param_string]
            
            # Get user and item representations using NMF
            data_path = '/Users/madisonthantu/Desktop/DREAM/data/ml-100k/u.data'
            if param_vals["compute_embeddings_via_surprise"]:
                user_representation, item_representation = compute_embeddings_surprise(data_path, separator="\t", num_attributes=param_vals["num_attrs"], num_epochs=args["num_epochs"])
            elif not param_vals["compute_embeddings_via_surprise"]:
                binary_ratings_matrix = load_and_process_movielens(file_path=data_path)
                user_representation, item_representation = compute_embeddings(binary_ratings_matrix, n_attrs=param_vals["num_attrs"], max_iter=args["max_iter"])
            else:
                raise Exception("❗️ Computing user and item representations failed ❗️")
            # Define topic clusters using NMF
            item_cluster_ids, item_cluster_centers = compute_constrained_clusters(embeddings=item_representation.T, name='item_clusters', n_clusters=param_vals["num_clusters"])
            user_cluster_ids, user_cluster_centers = compute_constrained_clusters(embeddings=user_representation, name='user_clusters', n_clusters=param_vals["num_clusters"])
            global_user_cluster_ids, global_user_cluster_centers = compute_constrained_clusters(embeddings=user_cluster_centers, name='global_user_clusters', n_clusters=1)
            # Get user pairs - global user pairs, intra-cluster user pairs, inter-cluster user pairs
            global_user_pairs = create_global_user_pairs(user_cluster_ids)
            if param_vals["create_cluster_user_pairs_by_user_topic_mapping"]:
                print("Created user pairs via user-topic mapping")
                user_item_cluster_mapping = user_topic_mapping(user_representation, item_cluster_centers)
                inter_cluster_user_pairs, intra_cluster_user_pairs = create_cluster_user_pairs(user_item_cluster_mapping)
            elif not param_vals["create_cluster_user_pairs_by_user_topic_mapping"]:
                print("Created user pairs via user cluster IDs")
                inter_cluster_user_pairs, intra_cluster_user_pairs = create_cluster_user_pairs(user_cluster_ids)
            else:
                raise Exception("❗️ Creating cluster user pairs failed ❗️")
            
            metrics_list = [
                MSEMeasurement(diagnostics=True),  
                InteractionSpread(),                
                InteractionSimilarity(pairs=global_user_pairs, name='global_interaction_similarity', diagnostics=True), 
                InteractionSimilarity(pairs=inter_cluster_user_pairs, name='inter_cluster_interaction_similarity', diagnostics=True), 
                InteractionSimilarity(pairs=intra_cluster_user_pairs, name='intra_cluster_interaction_similarity', diagnostics=True), 
                MeanCosineSim(pairs=global_user_pairs, name='mean_global_cosine_sim', diagnostics=True),
                MeanCosineSim(pairs=intra_cluster_user_pairs, name='mean_intra_cluster_cosine_sim', diagnostics=True),
                MeanCosineSim(pairs=inter_cluster_user_pairs, name='mean_inter_cluster_cosine_sim', diagnostics=True),
                MeanCosineSimPerCluster(user_cluster_ids=user_cluster_ids, n_clusts=param_vals["num_clusters"], name="mean_cosine_sim_per_cluster", diagnostics=True), 
                MeanDistanceFromCentroid(user_cluster_ids=user_cluster_ids, user_centroids=user_cluster_centers, name="mean_cluster_distance_from_centroid", diagnostics=True), 
                MeanDistanceFromCentroid(user_cluster_ids=global_user_cluster_ids, user_centroids=global_user_cluster_centers, name="mean_global_distance_from_centroid", diagnostics=True), 
                MeanDistanceFromCentroidPerCluster(user_cluster_ids=user_cluster_ids, user_centroids=user_cluster_centers, n_clusts=param_vals["num_clusters"], name="mean_distance_from_centroid_per_cluster", diagnostics=True),
                InteractionMeasurement(name="interaction_histogram"),
                RMSEMeasurement(),
                NoveltyMetric(diagnostics=True),
                DiversityMetric(),
                TopicInteractionMeasurement(),
                TopicInteractionSpread(),
                UserMSEMeasurement()
            ]
            
            model = run_bubble_burster(
                user_representation=user_representation, 
                item_representation=item_representation, 
                item_cluster_ids=item_cluster_ids, 
                metrics_list=metrics_list,
                param_dict=param_vals,
                args=args
            )
            # Saving final metrics
            for metric in metrics_list:
                result_metrics[metric.name][param_string].append(process_measurement(model, metric.name))
                # Recording model diagnostics
                if metric.name in diagnostic_metrics:
                    for diagnostic in diagnostics_vars:
                        model_diagnostics[metric.name][diagnostic].append(process_diagnostic(metric, diagnostic))
            # Saving simulation environment variables
            sim_environment["actual_user_representation_initial"][param_string].append(user_representation)
            sim_environment["actual_user_representation_final"][param_string].append(model.users.actual_user_profiles.value)
            sim_environment["user_cluster_assignments"][param_string].append(user_cluster_ids)
            sim_environment["user_cluster_centroids"][param_string].append(user_cluster_centers)
            sim_environment["item_representation"][param_string].append(item_representation)
            sim_environment["item_cluster_assignments"][param_string].append(item_cluster_ids)
            sim_environment["item_cluster_centroids"][param_string].append(item_cluster_centers)
            sim_environment["global_user_centroid"][param_string].append(global_user_cluster_centers)
            if param_vals["create_cluster_user_pairs_by_user_topic_mapping"]:
                sim_environment["user_item_cluster_mapping"][param_string].append(user_item_cluster_mapping)
            else:
                sim_environment["user_item_cluster_mapping"][param_string].append(np.array([]))
                
            for metric in result_diagnostics:
                result_diagnostics[metric][param_string] = model_diagnostics[metric]
        
    # write results to pickle file
    output_file_metrics = os.path.join(args["output_dir"], "sim_results.pkl")
    pkl.dump(result_metrics, open(output_file_metrics, "wb"), -1)
    
    output_file_diagnostics = os.path.join(args["output_dir"], "sim_diagnostics.pkl")
    pkl.dump(result_diagnostics, open(output_file_diagnostics, "wb"), -1)
    
    output_file_sim_env = os.path.join(args["output_dir"], "sim_environment.pkl")
    pkl.dump(sim_environment, open(output_file_sim_env, "wb"), -1)
    print("Done with simulation! 🎉")