import numpy as np
from surprise import Dataset, Reader, NMF
from surprise.accuracy import mse
from collections import defaultdict
import pickle as pkl
import os


def compute_embeddings_surprise(file_path, separator, num_attributes, num_epochs=50):
    reader = Reader(line_format="user item rating timestamp", sep=separator)
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    print('Calculating embeddings via `surprise` ...')
    algo = NMF(n_factors=num_attributes, n_epochs=num_epochs)
    algo.fit(trainset)
    print(f"... Calculated embeddings.")
    # pu -> user factors
    # qi -> item factors
    user_embedding, item_embedding = algo.pu, algo.qi
    return user_embedding, item_embedding.T


def load_sim_results(folder, filename="sim_results.pkl"):
    filepath = os.path.join(folder, filename)
    return pkl.load(open(filepath, "rb"))


def merge_results(folder_paths, filenames=None):
    """
    Paths must be paths to pickle files resulting from multiple trials of the same
    simulation setup
    """
    final_result = defaultdict(lambda: defaultdict(list))

    for idx in range(len(folder_paths)):
        if filenames is not None:
            results = load_sim_results(folder_paths[idx], filenames[idx])
        else:
            results = load_sim_results(folder_paths[idx])
        # key = metric, value = dictionary mapping algo name to list of entries
        for metric_name, v in results.items():
            for model_name, metric_vals in v.items():
                # merge the list
                final_result[metric_name][model_name] += metric_vals

    return final_result