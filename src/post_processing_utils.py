import os

import numpy as np
import pandas as pd

# user cluster assignments, user cluster centroids
# item cluster assignments, item cluster centroids
# actual_user_representation_initial, actual_user_representation_final
# user & item embeddings (via NMF)
# user_item_cluster_mapping

def load_or_create_measurements_df(model, model_name, train_timesteps, file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
    else:
        measurements = model.get_measurements()
        df = pd.DataFrame(measurements)
        df['state'] = 'train' # makes it easier to later understand which part was training
        df.loc[df['timesteps'] > train_timesteps, 'state'] = 'run'
        df['model'] = model_name
    
    return df


def collect_parameters(file, columns):   
    file_name = file[:-4]
    params = file_name.split('_')
    params_start_id = params.index('measurements')
    row = {}
    row['model_name'] = '_'.join(params[:params_start_id])
    for col in columns:
        for param in params:
            if param.endswith(col):
                value = param[:param.find(col)]
                row[col] = value
    return row


def load_measurements(path, numeric_columns):
    dfs = []
    data = []
    columns = ['model_name'] + numeric_columns
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            row = collect_parameters(file, columns)
            data.append(row)
            df = pd.read_csv(path + '/' + file)
            dfs.append(df)
    
    parameters_df = pd.DataFrame().append(data)
    for col in numeric_columns:
        parameters_df[col] = pd.to_numeric(parameters_df[col])
    return dfs, parameters_df


def create_parameter_string(naming_config):
    parameters_str = ''
    for key, value in naming_config.items():
        parameters_str += f'_{value}{key}'
        

def process_diagnostic(metric, diagnostics_var):
    return metric.get_diagnostics()[diagnostics_var].to_numpy()