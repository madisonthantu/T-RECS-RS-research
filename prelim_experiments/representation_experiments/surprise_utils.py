import numpy as np
from surprise import Dataset, Reader, NMF
from surprise.accuracy import mse


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