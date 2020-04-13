# dbscan clustering
from numpy import unique
from numpy import where
import numpy as np
import pickle

from sklearn.datasets import make_classification
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

import pandas as pd
from matplotlib import pyplot as plt

def preprocessor(df):
    df = df.astype('float32',copy=False)
    #Small preprocessing Pipeline
    stscaler = StandardScaler().fit(df)
    df = stscaler.transform(df)
    return df
    
def train(dataset,eps):
    data = pd.read_csv(dataset)
    data.fillna(method = 'ffill',inplace=True)
    IMAGE_FOLDER = 'static//'
    MODEL_FOLDER = 'models//'

    header = data.columns.values
    plt.figure(0)
    

    plt.scatter(data[header[0]],data[header[1]])
    unlabelled_data = IMAGE_FOLDER+'optics_unlabelled.png'
    plt.savefig(unlabelled_data)

    df = preprocessor(data)

    model = OPTICS(eps=eps,min_samples=9)

    yhat = model.fit_predict(df)
    # retrieve unique clusters
    clusters = unique(yhat)

    plt.figure(1)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(df[row_ix, 0], df[row_ix, 1])
    labelled_data = IMAGE_FOLDER+'optics.png'
    plt.savefig(labelled_data)
    # show the plot

    return ('Model Graphically Represented') 
    
    
