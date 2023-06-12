import sys
sys.path.insert(1, '/Users/madisonthantu/Desktop/DREAM/t-recs')
from trecs.metrics import MSEMeasurement, InteractionSpread, InteractionSpread, InteractionSimilarity, RecSimilarity, RMSEMeasurement, InteractionMeasurement
from trecs.components import Users
import trecs.matrix_ops as mo
from trecs.models import ContentFiltering
from collections import defaultdict
import pickle as pkl
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np


# utility function to extract measurement
def process_measurement(model, metric_string):
    # get rid of the None value at the beginning
    return model.get_measurements()[metric_string][1:]


def load_sim_results(folder, filename="sim_results.pkl"):
    filepath = os.path.join(folder, filename)
    return pkl.load(open(filepath, "rb"))


def merge_results(folder_paths):
    """
    Paths must be paths to pickle files resulting from multiple trials of the same
    simulation setup
    """
    final_result = defaultdict(lambda: defaultdict(list))

    for folder_path in folder_paths:
        results = load_sim_results(folder_path)
        # key = metric, value = dictionary mapping algo name to list of entries
        for metric_name, v in results.items():
            for model_name, metric_vals in v.items():
                # merge the list
                final_result[metric_name][model_name] += metric_vals

    return final_result


"""
Graphing results utilities
"""
def transform_relative_to_ideal(train_results, metric_key, model_keys, absolute_measure=True):
    relative_dist = defaultdict(lambda: defaultdict(list))

    if absolute_measure:
        ideal_dist = np.array(train_results[metric_key]["ideal"])
    else:
        model_key = list(train_results[metric_key].keys())[0]
        # zeros for all timsteps
        trials = len(train_results[metric_key][model_key])
        timesteps = len(train_results[metric_key][model_key][0])
        ideal_dist = np.zeros((trials, timesteps))
        relative_dist[metric_key]["ideal"] = ideal_dist

    for model_key in model_keys:
        if model_key is "ideal" and not absolute_measure:
            continue

        abs_dist = np.array(train_results[metric_key][model_key])
        if absolute_measure:
            abs_dist = abs_dist - ideal_dist
        relative_dist[metric_key][model_key] = abs_dist
    return relative_dist


def graph_metrics(train_results, metric_key, model_keys, label_map, mean_sigma=0, mult_sd=0, conf_sigma=0):
    for m in model_keys:
        if not isinstance(train_results[metric_key][m], np.ndarray):
            train_results[metric_key][m] = np.array(train_results[metric_key][m])
        # average across trials and smooth, if necessary
        if mean_sigma > 0:
            values = gaussian_filter1d(train_results[metric_key][m].mean(axis=0), sigma=mean_sigma)
        else:
            values = train_results[metric_key][m].mean(axis=0)
        line = plt.plot(values, label=label_map[m])
        line_color = line[0].get_color()
        if mult_sd > 0:
            std = train_results[metric_key][m].std(axis=0)
            timesteps = np.arange(len(std))
            low = values - mult_sd * std
            high = values + mult_sd * std
            if conf_sigma > 0:
                low = gaussian_filter1d(low, sigma=conf_sigma)
                high = gaussian_filter1d(high, sigma=conf_sigma)
            plt.fill_between(timesteps, low, high, color = line_color, alpha=0.3)
    plt.legend(facecolor='white', framealpha=1, loc='upper right', bbox_to_anchor=(1.7, 1.0))


def graph_relative_to_ideal(train_results, metric_key, model_keys, label_map, absolute_measure=True, mean_sigma=0, mult_sd=0, conf_sigma=0):
    relative_dist = transform_relative_to_ideal(train_results, metric_key, model_keys, absolute_measure)
    graph_metrics(relative_dist, metric_key, model_keys, label_map, mean_sigma, mult_sd, conf_sigma)
    
    
def graph_metrics_by_axis(ax, train_results, metric_key, model_keys, label_map, mean_sigma=0, mult_sd=0, conf_sigma=0):
    for m in model_keys:
        if not isinstance(train_results[metric_key][m], np.ndarray):
            train_results[metric_key][m] = np.array(train_results[metric_key][m])
        # average across trials and smooth, if necessary
        if mean_sigma > 0:
            values = gaussian_filter1d(train_results[metric_key][m].mean(axis=0), sigma=mean_sigma)
        else:
            values = train_results[metric_key][m].mean(axis=0)
        line = ax.plot(values, label=label_map[m])
        line_color = line[0].get_color()
        if mult_sd > 0:
            std = train_results[metric_key][m].std(axis=0)
            timesteps = np.arange(len(std))
            low = values - mult_sd * std
            high = values + mult_sd * std
            if conf_sigma > 0:
                low = gaussian_filter1d(low, sigma=conf_sigma)
                high = gaussian_filter1d(high, sigma=conf_sigma)
            ax.fill_between(timesteps, low, high, color = line_color, alpha=0.3)
    ax.legend(facecolor='white', framealpha=1, loc='upper right', bbox_to_anchor=(1, 0.5))
    return ax


def transform_relative_to_global(train_results, global_metric_key, metric_key, model_keys, absolute_measure=True):
    proprtional_dist = defaultdict(lambda: defaultdict(list))
    
    for model_key in model_keys:
        global_dist = np.array(train_results[global_metric_key][model_key])
        metric_dist = np.array(train_results[metric_key][model_key])
        proprtional_dist[metric_key][model_key] = np.divide(metric_dist, global_dist)
        
    return proprtional_dist


def graph_relative_to_global_by_axis(ax, train_results, global_metric_key, metric_key, model_keys, label_map, absolute_measure=True, mean_sigma=0, mult_sd=0, conf_sigma=0):
    relative_dist = transform_relative_to_global(train_results, global_metric_key, metric_key, model_keys, absolute_measure)
    graph_metrics_by_axis(ax, relative_dist, metric_key, model_keys, label_map, mean_sigma, mult_sd, conf_sigma)