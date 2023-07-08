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

# import inspect module
import inspect


def run_bubble_burster(
    user_representation, 
    item_representation,
    item_cluster_ids,
    metrics_list,
    user_config,
    model_config,
    train_config,
    run_config
    ):
    
    users = Users(
        actual_user_profiles=user_representation, 
        **user_config
    )
    print()
    for k, v in vars(users).items():
        print(k, v)
    print()
    bubble = BubbleBurster(
        actual_user_representation=users, 
        actual_item_representation=item_representation,
        item_topics=item_cluster_ids,
        **model_config
    )
    for k, v in vars(bubble).items():
        print(k, v)
    print()
    bubble.add_metrics(*metrics_list)
    bubble.startup_and_train(**train_config)
    bubble.run(**run_config)
    bubble.close() # end logging
    return bubble


def create_output_string(args):
    model_name = f"{args['model_name']}-{args['repeated_training']}retraining-{args['probabilistic']}probabilistic"
    # model_name = f"{args['model_name']}-{args['repeated_training']}retraining-{args['probabilistic']}probabilistic-{args['repeated_items_repeat_interactions']}repeated_items_repeat_interactions"
    if args['random_items_per_iter']:
        model_name += f"-{args['random_items_per_iter']}random_items_per_iter"
    if args['repeated_items_repeat_interactions']:
        model_name += f"-{args['repeated_items_repeat_interactions']}repeated_items_repeat_interactions"
    if args['score_fn']:
        model_name += f"-{args['score_fn']}_score_fn"
    if args['interleaving_fn']:
        model_name += f"-{args['interleaving_fn']}_interleaving_fn"
    return model_name


"""
DONE:
[1][a][i]   Baseline, myopic, single training, not probabilistic, user repeat interactions not allowed and recommender repeated items not allowed
            python run_sim_experiments.py  --output_dir sim_results  --model_name baseline_myopic  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 5  --probabilistic 0
[1][a][ii]  Baseline, myopic, repeated training, not probabilistic, user repeat interactions are allowed and recommender repeated items are allowed
            python run_sim_experiments.py  --output_dir sim_results  --model_name baseline_myopic  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 5  --probabilistic 0  --repeated_items_repeat_interactions 1

[1][b][i]   Baseline, myopic, repeated training, not probabilistic, user repeat interactions not allowed and recommender repeated items not allowed
            python run_sim_experiments.py  --output_dir sim_results  --model_name baseline_myopic  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 5  --probabilistic 0  --repeated_items_repeat_interactions 0
[1][b][ii]  Baseline, myopic, repeated training, not probabilistic, user repeat interactions are allowed and recommender repeated items are allowed
            python run_sim_experiments.py  --output_dir sim_results  --model_name baseline_myopic  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 5  --probabilistic 0  --repeated_items_repeat_interactions 1

[2][a]      Probabilistic recommender, single training
            python run_sim_experiments.py  --output_dir sim_results  --model_name probabilistic_recommender  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 5  --probabilistic 1
[2][b]      Probabilistic recommender, repeated training 
            python run_sim_experiments.py  --output_dir sim_results  --model_name probabilistic_recommender  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 5  --probabilistic 1

[3][a]      Random recommender (# random_items_in_slate == # items_in_slate), single training
            python run_sim_experiments.py  --output_dir sim_results  --model_name random_recommender  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 5  --random_items_per_iter 10
[3][b]      Random recommender (# random_items_in_slate == # items_in_slate), repeated training
            python run_sim_experiments.py  --output_dir sim_results  --model_name random_recommender  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 5  --random_items_per_iter 10

TO DO:
[4][a]      Random interleaving recommender (# random_items_in_slate == # items_in_slate), single training
            python run_sim_experiments.py  --output_dir sim_results  --model_name random_interleaving  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 5  --random_items_per_iter 5
[4][b]      Random interleaving recommender (# random_items_in_slate == # items_in_slate), repeated training
            python run_sim_experiments.py  --output_dir sim_results  --model_name random_interleaving  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 5  --random_items_per_iter 5


"""
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='running parameter experiments')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    
    parser.add_argument('--drift', type=int, default=0.1)
    parser.add_argument('--attention_exp', type=int, default=-0.8)
    parser.add_argument('--num_attrs', type=int, default=20)
    parser.add_argument('--num_clusters', type=int, default=15)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--create_cluster_user_pairs_by_user_topic_mapping', type=int, default=True)
    
    parser.add_argument('--repeated_training', type=int, default=True)
    parser.add_argument('--num_sims', type=int, default=5)
    parser.add_argument('--startup_iters', type=int, required=True)
    parser.add_argument('--sim_iters', type=int, required=True)
    parser.add_argument('--seed', type=int, default=random_state)
    
    parser.add_argument('--probabilistic', type=int, default=0)
    parser.add_argument('--random_items_per_iter', type=int, default=0)
    parser.add_argument('--repeated_items_repeat_interactions', type=int,  default=0)
    
    parser.add_argument('--score_fn', type=str,  default='')
    parser.add_argument('--interleaving_fn', type=str,  default='')
    

    parsed_args = parser.parse_args()
    args = vars(parsed_args)
    if args["repeated_training"] == False:
        assert(args["startup_iters"] == args["sim_iters"]), "Incorrect ratio of startup to sim iters supplied for repeated_training=False"
    else:
        assert(args["startup_iters"] < (args["sim_iters"]/4)), "Incorrect ratio of startup to sim iters supplied for repeated_training=True"
    
    model_name = create_output_string(args)
    if args["repeated_training"]:
        output_directory = f"{args['output_dir']}/repeated_training/{model_name}"
    else:
        output_directory = f"{args['output_dir']}/single_training/{model_name}"
    if os.path.isdir(output_directory):
        print(f"The supplied output directory, {[output_directory]}, already exists. Do you wish to overwrite this directory's contents? [y/n]: ")
        if str(input()).lower() != "y":
            sys.exit()
        
    print("Creating experiment output folder... 💻")
    # create output folder
    try:
        os.makedirs(output_directory)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    users_config = {
        'repeat_interactions':args["repeated_items_repeat_interactions"],
        'drift':args["drift"],
        'attention_exp':args["attention_exp"]
    }
    model_config = {
        'num_attributes': args["num_attrs"],
        'num_items_per_iter': 10,
        'record_base_state': True,
        'probabilistic_recommendations':args["probabilistic"]
    }
    # if not args["score_fn"]:
    #     model_config["score_fn"] = None
    # if not args["interleaving_fn"]:
    #     model_config["interleaving_fn"] = None
    train_config = {
        'timesteps':args["startup_iters"], 
        'no_new_items':True
    }    
    run_config = {
        'timesteps':args["sim_iters"], 
        'train_between_steps':args["repeated_training"], 
        'repeated_items':args["repeated_items_repeat_interactions"], 
        'random_items_per_iter':args["random_items_per_iter"],
        'no_new_items':True
    }

    # write experiment arguments to file
    with open(os.path.join(output_directory, "args.txt"), "w") as log_file:
        pprint.pprint(model_name, log_file)
        pprint.pprint("SCRIPT ARGS", log_file)
        pprint.pprint(args, log_file)
        pprint.pprint("USER CONFIG", log_file)
        pprint.pprint(users_config, log_file)
        pprint.pprint("MODEL CONFIG", log_file)
        pprint.pprint(model_config, log_file)
        pprint.pprint("RUN SIM CONFIG", log_file)
        pprint.pprint(run_config, log_file)

    rng = np.random.default_rng(args["seed"])

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
    data_path = '/Users/madisonthantu/Desktop/DREAM/data/ml-100k/u.data'
    
    for sim_num in range(args["num_sims"]):
        print(f"Simulation [{sim_num+1}/{args['num_sims']}]: {model_name}")
        # Get user and item representations using NMF
        binary_ratings_matrix = load_and_process_movielens(file_path=data_path)
        user_representation, item_representation = compute_embeddings(binary_ratings_matrix, n_attrs=args["num_attrs"], max_iter=args["max_iter"])
        # Define topic clusters using NMF
        item_cluster_ids, item_cluster_centers = compute_constrained_clusters(embeddings=item_representation.T, name='item_clusters', n_clusters=args["num_clusters"])
        user_cluster_ids, user_cluster_centers = compute_constrained_clusters(embeddings=user_representation, name='user_clusters', n_clusters=args["num_clusters"])
        global_user_cluster_ids, global_user_cluster_centers = compute_constrained_clusters(embeddings=user_cluster_centers, name='global_user_clusters', n_clusters=1)
        # Get user pairs - global user pairs, intra-cluster user pairs, inter-cluster user pairs
        global_user_pairs = create_global_user_pairs(user_cluster_ids)
        user_item_cluster_mapping = user_topic_mapping(user_representation, item_cluster_centers)
        inter_cluster_user_pairs, intra_cluster_user_pairs = create_cluster_user_pairs(user_item_cluster_mapping)
            
        metrics_list = [
            MSEMeasurement(diagnostics=True),  
            InteractionSpread(),                
            InteractionSimilarity(pairs=global_user_pairs, name='global_interaction_similarity', diagnostics=True), 
            InteractionSimilarity(pairs=inter_cluster_user_pairs, name='inter_cluster_interaction_similarity', diagnostics=True), 
            InteractionSimilarity(pairs=intra_cluster_user_pairs, name='intra_cluster_interaction_similarity', diagnostics=True), 
            MeanCosineSim(pairs=global_user_pairs, name='mean_global_cosine_sim', diagnostics=True),
            MeanCosineSim(pairs=intra_cluster_user_pairs, name='mean_intra_cluster_cosine_sim', diagnostics=True),
            MeanCosineSim(pairs=inter_cluster_user_pairs, name='mean_inter_cluster_cosine_sim', diagnostics=True),
            MeanCosineSimPerCluster(user_cluster_ids=user_cluster_ids, n_clusts=args["num_clusters"], name="mean_cosine_sim_per_cluster", diagnostics=True), 
            MeanDistanceFromCentroid(user_cluster_ids=user_cluster_ids, user_centroids=user_cluster_centers, name="mean_cluster_distance_from_centroid", diagnostics=True), 
            MeanDistanceFromCentroid(user_cluster_ids=global_user_cluster_ids, user_centroids=global_user_cluster_centers, name="mean_global_distance_from_centroid", diagnostics=True), 
            MeanDistanceFromCentroidPerCluster(user_cluster_ids=user_cluster_ids, user_centroids=user_cluster_centers, n_clusts=args["num_clusters"], name="mean_distance_from_centroid_per_cluster", diagnostics=True),
            InteractionMeasurement(name="interaction_histogram"),
            RMSEMeasurement(),
            NoveltyMetric(diagnostics=True),
            DiversityMetric(),
            TopicInteractionMeasurement(),
            TopicInteractionSpread(),
            UserMSEMeasurement()
        ]
        assert(len(metric_names) == len(metrics_list)), "Metric names and metrics list do not match up"
            
        model = run_bubble_burster(
            user_representation=user_representation, 
            item_representation=item_representation, 
            item_cluster_ids=item_cluster_ids, 
            metrics_list=metrics_list,
            user_config=users_config,
            model_config=model_config,
            train_config=train_config,
            run_config=run_config
        )
        # Saving final metrics
        for metric in metrics_list:
            result_metrics[metric.name][model_name].append(process_measurement(model, metric.name))
            # Recording model diagnostics
            if metric.name in diagnostic_metrics:
                for diagnostic in diagnostics_vars:
                    model_diagnostics[metric.name][diagnostic].append(process_diagnostic(metric, diagnostic))
                        
            # Saving simulation environment variables
            sim_environment["actual_user_representation_initial"][model_name].append(user_representation)
            sim_environment["actual_user_representation_final"][model_name].append(model.users.actual_user_profiles.value)
            sim_environment["user_cluster_assignments"][model_name].append(user_cluster_ids)
            sim_environment["user_cluster_centroids"][model_name].append(user_cluster_centers)
            sim_environment["item_representation"][model_name].append(item_representation)
            sim_environment["item_cluster_assignments"][model_name].append(item_cluster_ids)
            sim_environment["item_cluster_centroids"][model_name].append(item_cluster_centers)
            sim_environment["global_user_centroid"][model_name].append(global_user_cluster_centers)
            sim_environment["user_item_cluster_mapping"][model_name].append(user_item_cluster_mapping)
                
    for metric in result_diagnostics:
        result_diagnostics[metric][model_name] = model_diagnostics[metric]
        
    # write results to pickle file
    output_file_metrics = os.path.join(output_directory, "sim_results.pkl")
    pkl.dump(result_metrics, open(output_file_metrics, "wb"), -1)
    
    output_file_diagnostics = os.path.join(output_directory, "sim_diagnostics.pkl")
    pkl.dump(result_diagnostics, open(output_file_diagnostics, "wb"), -1)
    
    output_file_sim_env = os.path.join(output_directory, "sim_environment.pkl")
    pkl.dump(sim_environment, open(output_file_sim_env, "wb"), -1)
    
    print("Done with simulation! 🎉")