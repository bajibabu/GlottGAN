import os
import numpy as np
from scipy.io import netcdf
from sklearn import preprocessing
from sklearn.externals import joblib

def load_data(data_dir, num_files=30):
    files_list = os.listdir(data_dir)
    dataset = []
    ac_dataset = []
    for fname in files_list[:num_files]:
        #print(fname)
        f = os.path.join(data_dir, fname)
        with netcdf.netcdf_file(f, 'r') as fid:
            m = fid.variables['outputMeans'][:].copy()
            s = fid.variables['outputStdevs'][:].copy()
            feats = fid.variables['targetPatterns'][:].copy()
            ac_feats = fid.variables['inputs'][:].copy()
            scaler = preprocessing.StandardScaler()
            scaler.mean_ = m
            scaler.scale_ = s
            feats = scaler.inverse_transform(feats)
            assert feats.shape[0] == ac_feats.shape[0]
            dataset.extend(feats)
            ac_dataset.extend(ac_feats)
    dataset = np.asarray(dataset)
    ac_dataset = np.asarray(ac_dataset)
    #print(dataset.shape, ac_dataset.shape)
    return dataset, ac_dataset