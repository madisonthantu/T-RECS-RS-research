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
from wrapper.models.xQuAD import xQuAD
from wrapper.metrics.evaluation_metrics import DiversityMetric, NoveltyMetric, TopicInteractionMeasurement, TopicInteractionSpread, UserMSEMeasurement
from wrapper.metrics.clustering_metrics import MeanCosineSim, MeanDistanceFromCentroid, MeanCosineSimPerCluster, MeanDistanceFromCentroidPerCluster, UserDistanceFromClusterCentroid
from src.utils import compute_constrained_clusters, create_global_user_pairs, user_topic_mapping, create_cluster_user_pairs, load_and_process_movielens, compute_embeddings
from src.post_processing_utils import process_diagnostic

import argparse
import errno
import warnings
import pprint
import pickle as pkl
warnings.simplefilter("ignore")

# import inspect module
import inspect

random_state = np.random.RandomState()

def get_metrics(args):
    metrics = [
        MSEMeasurement(diagnostics=True),  
        InteractionSpread(),                
        InteractionSimilarity(pairs=global_user_pairs, name='global_interaction_similarity'), 
        InteractionSimilarity(pairs=inter_cluster_user_pairs, name='inter_cluster_interaction_similarity'), 
        InteractionSimilarity(pairs=intra_cluster_user_pairs, name='intra_cluster_interaction_similarity'), 
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
        UserMSEMeasurement(),
        UserDistanceFromClusterCentroid(user_cluster_ids=user_cluster_ids, user_centroids=user_cluster_centers, n_clusts=args["num_clusters"], name="user_distance_from_cluster_centroid", verbose=False)
    ]
    return metrics

def run_baseline_myopic_recommender(users_config, model_config, train_config, run_config, args):
    users = Users(
        **users_config,
        repeat_interactions=0,  # ********** #
    )
    baseline_myopic = BubbleBurster(
        actual_user_representation=users, 
        **model_config,
        probabilistic_recommendations=0     # ********** #
    )
    baseline_myopic.add_metrics(*get_metrics(args))
    baseline_myopic.startup_and_train(**train_config)
    baseline_myopic.run(
        **run_config,
        repeated_items=0,       # ********** #
        random_items_per_iter=0 # ********** #
    )
    baseline_myopic.close() # end logging
    return baseline_myopic

def run_repeated_items_repeat_interactions_recommender(users_config, model_config, train_config, run_config, args):
    users = Users(
        **users_config,
        repeat_interactions=1,  # ********** #
    )
    repeated = BubbleBurster(
        actual_user_representation=users, 
        **model_config,
        probabilistic_recommendations=0     # ********** #
    )
    repeated.add_metrics(*get_metrics(args))
    repeated.startup_and_train(**train_config)
    repeated.run(
        **run_config,
        repeated_items=1,       # ********** #
        random_items_per_iter=0 # ********** #
    )
    repeated.close() # end logging
    return repeated


def run_probabilistic_recommender(users_config, model_config, train_config, run_config, args):
    users = Users(
        **users_config,
        repeat_interactions=0,  # ********** #
    )
    probabilistic = BubbleBurster(
        actual_user_representation=users, 
        **model_config,
        probabilistic_recommendations=1     # ********** #
    )
    probabilistic.add_metrics(*get_metrics(args))
    probabilistic.startup_and_train(**train_config)
    probabilistic.run(
        **run_config,
        repeated_items=0,       # ********** #
        random_items_per_iter=0 # ********** #
    )
    probabilistic.close() # end logging
    return probabilistic

def run_random_recommender(users_config, model_config, train_config, run_config, args):
    users = Users(
        **users_config,
        repeat_interactions=0,  # ********** #
    )
    random = BubbleBurster(
        actual_user_representation=users, 
        **model_config,
        probabilistic_recommendations=0     # ********** #
    )
    random.add_metrics(*get_metrics(args))
    random.startup_and_train(**train_config)
    random.run(
        **run_config,
        repeated_items=0,       # ********** #
        random_items_per_iter=10 # ********** #
    )
    random.close() # end logging
    return random


def run_random_interleaving_recommender(users_config, model_config, train_config, run_config, args):
    users = Users(
        **users_config,
        repeat_interactions=0,  # ********** #
    )
    interleave = BubbleBurster(
        actual_user_representation=users, 
        **model_config,
        probabilistic_recommendations=0     # ********** #
    )
    interleave.add_metrics(*get_metrics(args))
    interleave.startup_and_train(**train_config)
    interleave.run(
        **run_config,
        repeated_items=0,       # ********** #
        random_items_per_iter=4 # ********** #
    )
    interleave.close() # end logging
    return interleave
    

def run_xquad_recommender(users_config, model_config, train_config, run_config, args, method, alpha):
    users = Users(
        **users_config,
        repeat_interactions=0,  # ********** #
    )
    xquad = xQuAD(
        actual_user_representation=users, 
        xquad_method=method,
        alpha=alpha,
        **model_config,
        probabilistic_recommendations=0     # ********** #
    )
    xquad.add_metrics(*get_metrics(args))
    xquad.startup_and_train(**train_config)
    xquad.run(
        **run_config,
        repeated_items=0,       # ********** #
        random_items_per_iter=0 # ********** #
    )
    xquad.close() # end logging
    return xquad


def save_model_results(model_key, model, result_metrics, result_diagnostics, result_environment, sim_num):
    model_metrics = model.metrics
    for metric in model_metrics:
        result_metrics[str(metric.name)][model_key].append(process_measurement(model, metric.name))
        # Recording model diagnostics
        if metric.name in diagnostic_metrics:
            for diagnostic in diagnostics_vars:
                if sim_num == 0:
                    curr_diagnostics = {k: list for k in diagnostics_vars}
                    for diagnostic in diagnostics_vars:
                        curr_diagnostics[diagnostic] = process_diagnostic(metric, diagnostic)
                else:
                    curr_diagnostics = result_diagnostics[metric.name][model_key]
                    for diagnostic in diagnostics_vars:
                        curr_diagnostics[diagnostic].append((process_diagnostic(metric, diagnostic)))
                result_diagnostics[metric.name][model_key] = curr_diagnostics
    # Saving simulation environment variables
    result_environment["actual_user_representation_initial"][model_key].append(user_representation)
    result_environment["actual_user_representation_final"][model_key].append(model.users.actual_user_profiles.value)
    result_environment["user_cluster_assignments"][model_key].append(user_cluster_ids)
    result_environment["user_cluster_centroids"][model_key].append(user_cluster_centers)
    result_environment["item_representation"][model_key].append(item_representation)
    result_environment["item_cluster_assignments"][model_key].append(item_cluster_ids)
    result_environment["item_cluster_centroids"][model_key].append(item_cluster_centers)
    result_environment["global_user_centroid"][model_key].append(global_user_cluster_centers) 
    result_environment["user_item_cluster_mapping"][model_key].append(user_item_cluster_mapping)
    
    return result_metrics, result_diagnostics, result_environment
        
"""
-   User pairs created via user-topic mapping, num_clusters=15
    python run_sim_experiments_full.py  --output_dir all_sim_results/simulation1  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 1
    python run_sim_experiments_full.py  --output_dir all_sim_results/simulation1  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 1
-   User pairs created via user clusters, num_clusters=15
    python run_sim_experiments_full.py  --output_dir all_sim_results/user_pairs_via_user_clusters/simulation1  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 1
    
* * * * * * * * *   
 ____________
| DONE  ...  |
 ————————————
 
Creating user pairs via user clusters, num_clusters=10
    -   seed=42
        -   [Done]  python run_sim_experiments_full.py  --output_dir all_sim_results/user_pairs_via_user_clusters/10clusters/simulation1  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 1
        -   [Done]  python run_sim_experiments_full.py  --output_dir all_sim_results/user_pairs_via_user_clusters/10clusters/simulation1  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 1

Creating user pairs via user clusters, num_clusters=15 
    -   Simulation 1
        -   [Done]  python run_sim_experiments_full.py  --output_dir all_sim_results/user_pairs_via_user_clusters/15clusters/simulation1  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 1
        -   [Done]  python run_sim_experiments_full.py  --output_dir all_sim_results/user_pairs_via_user_clusters/15clusters/simulation1  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 1 
    -   Simulation 2
        -   [Done]  python run_sim_experiments_full.py  --output_dir all_sim_results/user_pairs_via_user_clusters/15clusters/simulation2  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 1
    *** -   [Not]   python run_sim_experiments_full.py  --output_dir all_sim_results/user_pairs_via_user_clusters/15clusters/simulation2  --seed 61 --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 1
        
Creating user pairs via user-topic mapping num_clusters=15 
    -   seed=42
        -   [Not]   python run_sim_experiments_full.py  --output_dir all_sim_results/user_topic_mapping/15clusters/simulation1 --seed 42  --repeated_training 1  --startup_iters 10  --sim_iters 90  --num_sims 1
    *** -   [Not]   python run_sim_experiments_full.py  --output_dir all_sim_results/user_topic_mapping/15clusters/simulation1 --seed 42  --repeated_training 0  --startup_iters 50  --sim_iters 50  --num_sims 1
    

 ____________
| IP    ...  |
 ————————————
 ____________
| TO DO ...  |
 ————————————
        
"""
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='running parameter experiments')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--repeated_training', type=int, default=True)
    parser.add_argument('--num_sims', type=int, default=5)
    parser.add_argument('--startup_iters', type=int, required=True)
    parser.add_argument('--sim_iters', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)

    parsed_args = parser.parse_args()
    args = vars(parsed_args)
    if args["repeated_training"] == False:
        assert(args["startup_iters"] == args["sim_iters"]), "Incorrect ratio of startup to sim iters supplied for repeated_training=False"
    else:
        assert(args["startup_iters"] <= (args["sim_iters"]/4)), "Incorrect ratio of startup to sim iters supplied for repeated_training=True"
        
    # if args["data"] != 'ml_100k':
    #     assert(args['data'] in args['output_dir']), "Data source must be included in output directory if not default data ml-100k"
    
    if args["repeated_training"]:
        output_directory = f"{args['output_dir']}/repeated_training"
    else:
        output_directory = f"{args['output_dir']}/single_training"
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
    
    with open(os.path.join(output_directory, "args.txt"), "w") as log_file:
        pprint.pprint(args, log_file)
    
    hyper_params = {
        "drift":0.1,
        "attention_exp":-0.8,
        # "num_clusters":10,
        "num_clusters":15,
        "num_attrs":20,
        "max_iter":1000
    }
    if "10" in args['output_dir']:
        assert(hyper_params['num_clusters'] == 10), "Are you SURE the correct number of clusters is supplied?"
    elif "15" in args['output_dir']:
        assert(hyper_params['num_clusters'] == 15), "Are you SURE the correct number of clusters is supplied?"

    metrics_list = [
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
        "mse_per_user",
        "user_distance_from_cluster_centroid"
    ]
    result_metrics = {k: defaultdict(list) for k in metrics_list}
    
    diagnostic_metrics = set((
        "mse",
        # "global_interaction_similarity",
        # "inter_cluster_interaction_similarity",
        # "intra_cluster_interaction_similarity",
        "mean_global_cosine_sim",
        "mean_intra_cluster_cosine_sim",
        "mean_inter_cluster_cosine_sim",
        "mean_cosine_sim_per_cluster",
        "mean_cluster_distance_from_centroid",
        "mean_global_distance_from_centroid",
        "mean_distance_from_centroid_per_cluster",
        "mean_novelty"
    ))
    diagnostics_vars = ["mean", "std", "median", "min", "max", "skew"]
    # metric_diagnostics = {k: defaultdict(list) for k in diagnostics_vars}
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
    
    train_config = {
        'timesteps':args["startup_iters"], 
    }    
    run_config = {
        'timesteps':args["sim_iters"], 
        'train_between_steps':args["repeated_training"], 
    }
    
    # model_keys = [
    #     "baseline_myopic", 
    #     "repeated_items_repeat_interactions", 
    #     "probabilistic", 
    #     "random", 
    #     "random_interleaving", 
    #     "xquad_binary_0.1",
    #     "xquad_binary_0.25",
    #     "xquad_smooth_0.1",
    #     "xquad_smooth_0.25"
    # ]
    
    print("Running simulations...👟")
    data_path = '/Users/madisonthantu/Desktop/DREAM/data/ml-100k/u.data'
    for sim_num in range(args["num_sims"]):
        print(f"Simulation [{sim_num+1}/{args['num_sims']}]")
        
        models = {} # temporarily stores models
        # Get user and item representations using NMF
        # if args['data'] == 'ml_1m':
        #     data_path = '/Users/madisonthantu/Desktop/DREAM/data/ml-1m/ratings.dat'
        #     binary_ratings_matrix = load_and_process_movielens_1M(file_path=data_path)
        # elif args['data'] == 'ml_100k':
        #     data_path = '/Users/madisonthantu/Desktop/DREAM/data/ml-100k/u.data'
        #     binary_ratings_matrix = load_and_process_movielens(file_path=data_path)
        binary_ratings_matrix = load_and_process_movielens(file_path=data_path)
        # Compute embedding vectors via NMF
        user_representation, item_representation = compute_embeddings(binary_ratings_matrix, n_attrs=hyper_params["num_attrs"], max_iter=hyper_params["max_iter"], seed=args['seed'])
        # Define user clusters and topic clusters via k-means
        item_cluster_ids, item_cluster_centers = compute_constrained_clusters(embeddings=item_representation.T, name='item_clusters', n_clusters=hyper_params["num_clusters"], seed=args['seed'])
        user_cluster_ids, user_cluster_centers = compute_constrained_clusters(embeddings=user_representation, name='user_clusters', n_clusters=hyper_params["num_clusters"], seed=args['seed'])
        global_user_cluster_ids, global_user_cluster_centers = compute_constrained_clusters(embeddings=user_cluster_centers, name='global_user_clusters', n_clusters=1, seed=args['seed'])
        # Get user pairs - global user pairs, intra-cluster user pairs, inter-cluster user pairs
        global_user_pairs = create_global_user_pairs(user_cluster_ids)
        
        print(user_representation[:2,:5])
        print(item_representation[:2,:5])
        print(item_cluster_ids[:5])
        print(user_cluster_ids[:5])
        sys.exit()
        
        user_item_cluster_mapping = user_topic_mapping(user_representation, item_cluster_centers)
        """
        print("Created user pairs via user-topic mapping")
        inter_cluster_user_pairs, intra_cluster_user_pairs = create_cluster_user_pairs(user_item_cluster_mapping)
        with open(os.path.join(output_directory, "args.txt"), "a") as log_file:
            pprint.pprint("Created user pairs via user-topic mapping", log_file)
        """
        print("Created user pairs via user cluster IDs")
        inter_cluster_user_pairs, intra_cluster_user_pairs = create_cluster_user_pairs(user_cluster_ids)
        with open(os.path.join(output_directory, "args.txt"), "a") as log_file:
            pprint.pprint("Created user pairs via user cluster IDs", log_file)
        
        users_config = {
            'actual_user_profiles':user_representation, 
            'drift':hyper_params["drift"],
            'attention_exp':hyper_params["attention_exp"]
        }
        model_config = {
            'num_attributes':hyper_params["num_attrs"],
            'num_items_per_iter': 10,
            'record_base_state': True,
            'actual_item_representation':item_representation,
            'item_topics':item_cluster_ids,
        }
        
        print("Running baseline_myopic...")
        # models["baseline_myopic"] = run_baseline_myopic_recommender(users_config, model_config, train_config, run_config, hyper_params)
        baseline_myopic = run_baseline_myopic_recommender(users_config, model_config, train_config, run_config, hyper_params)
        result_metrics, result_diagnostics, sim_environment = save_model_results("baseline_myopic", baseline_myopic, result_metrics, result_diagnostics, sim_environment, sim_num)
        
        print("Running repeated_items_repeat_interactions...")
        # models["repeated_items_repeat_interactions"] = run_repeated_items_repeat_interactions_recommender(users_config, model_config, train_config, run_config, hyper_params)
        repeated_items_repeat_interactions = run_repeated_items_repeat_interactions_recommender(users_config, model_config, train_config, run_config, hyper_params)
        result_metrics, result_diagnostics, sim_environment = save_model_results("repeated_items_repeat_interactions", repeated_items_repeat_interactions, result_metrics, result_diagnostics, sim_environment, sim_num)
        
        print("Running probabilistic...")
        # models["probabilistic"] = run_probabilistic_recommender(users_config, model_config, train_config, run_config, hyper_params)
        probabilistic = run_probabilistic_recommender(users_config, model_config, train_config, run_config, hyper_params)
        result_metrics, result_diagnostics, sim_environment = save_model_results("probabilistic", probabilistic, result_metrics, result_diagnostics, sim_environment, sim_num)
        
        print("Running random...")
        # models["random"] = run_random_recommender(users_config, model_config, train_config , run_config, hyper_params)
        random = run_random_recommender(users_config, model_config, train_config, run_config, hyper_params)
        result_metrics, result_diagnostics, sim_environment = save_model_results("random", random, result_metrics, result_diagnostics, sim_environment, sim_num)
        
        print("Running random_interleaving...")
        # models["random_interleaving"] = run_random_interleaving_recommender(users_config, model_config, train_config, run_config, hyper_params)
        random_interleaving = run_random_interleaving_recommender(users_config, model_config, train_config, run_config, hyper_params)
        result_metrics, result_diagnostics, sim_environment = save_model_results("random_interleaving", random_interleaving, result_metrics, result_diagnostics, sim_environment, sim_num)
        
        print("Running xquad_binary_0.1...")
        # models["xquad_binary_0.1"] = run_xquad_recommender(users_config, model_config, train_config, run_config, hyper_params, method='binary', alpha=0.1)
        xquad_binary_1 = run_xquad_recommender(users_config, model_config, train_config, run_config, hyper_params, method='binary', alpha=0.1)
        result_metrics, result_diagnostics, sim_environment = save_model_results("xquad_binary_0.1", xquad_binary_1, result_metrics, result_diagnostics, sim_environment, sim_num)
        
        print("Running xquad_binary_0.25...")
        # models["xquad_binary_0.25"] = run_xquad_recommender(users_config, model_config, train_config, run_config, hyper_params, method='binary', alpha=0.25)
        xquad_binary_25 = run_xquad_recommender(users_config, model_config, train_config, run_config, hyper_params, method='binary', alpha=0.25)
        result_metrics, result_diagnostics, sim_environment = save_model_results("xquad_binary_0.25", xquad_binary_25, result_metrics, result_diagnostics, sim_environment, sim_num)
        
        print("Running xquad_smooth_0.1...")
        # models["xquad_smooth_0.1"] = run_xquad_recommender(users_config, model_config, train_config, run_config, hyper_params, method='smooth', alpha=0.1)
        xquad_smooth_1 = run_xquad_recommender(users_config, model_config, train_config, run_config, hyper_params, method='smooth', alpha=0.1)
        result_metrics, result_diagnostics, sim_environment = save_model_results("xquad_smooth_0.1", xquad_smooth_1, result_metrics, result_diagnostics, sim_environment, sim_num)
        
        print("Running xquad_smooth_0.25...")
        # models["xquad_smooth_0.25"] = run_xquad_recommender(users_config, model_config, train_config, run_config, hyper_params, method='smooth', alpha=0.25)
        xquad_smooth_25 = run_xquad_recommender(users_config, model_config, train_config, run_config, hyper_params, method='smooth', alpha=0.25)
        result_metrics, result_diagnostics, sim_environment = save_model_results("xquad_smooth_0.25", xquad_smooth_25, result_metrics, result_diagnostics, sim_environment, sim_num)
        
    # write results to pickle file
    output_file_metrics = os.path.join(output_directory, "sim_results.pkl")
    pkl.dump(result_metrics, open(output_file_metrics, "wb"), -1)
    
    output_file_diagnostics = os.path.join(output_directory, "sim_diagnostics.pkl")
    pkl.dump(result_diagnostics, open(output_file_diagnostics, "wb"), -1)
    
    output_file_sim_env = os.path.join(output_directory, "sim_environment.pkl")
    pkl.dump(sim_environment, open(output_file_sim_env, "wb"), -1)
    
    print("Done with simulation! 🎉")