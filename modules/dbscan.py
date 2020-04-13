# dbscan clustering
from numpy import unique
from numpy import where
import numpy as np
import pickle

from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import pandas as pd
from matplotlib import pyplot as plt

def preprocessor(df):
    df = df.as_matrix().astype('float32',copy=False)
    #Small preprocessing Pipeline
    stscaler = StandardScaler().fit(df)
    df = stscaler.transform(df)
    return df
    
def train(dataset,eps):
    data = pd.read_csv(dataset)
    data.fillna(method = 'ffill',inplace=True)
    IMAGE_FOLDER = '..//images//'
    MODEL_FOLDER = 'models//'

    header = data.columns.values
    plt.figure(0)
    # plt.xlabel(data[0,0])
    # plt.ylabel(data[1,0])
    plt.scatter(data[header[0]],data[header[1]])
    plt.savefig(IMAGE_FOLDER+'dbscan_unlabelled_data'+'.png')

    df = preprocessor(data)

    model = DBSCAN(eps=eps,min_samples=9)

    # filename = MODEL_FOLDER  +'dbscan.sav'

    # pickle.dump(model, open(filename, 'wb'))

    yhat = model.fit_predict(df)
    # retrieve unique clusters
    clusters = unique(yhat)

    plt.figure(1)
    # plt.xlabel(df[0,0])
    # plt.ylabel(df[1,0])

    unlabelled_data = IMAGE_FOLDER+'dbsacn_unlabelled.png'
    plt.savefig(unlabelled_data)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(df[row_ix, 0], df[row_ix, 1])
    labelled_data = IMAGE_FOLDER+'dbscan'+'.png'
    plt.savefig(labelled_data)
    # show the plot

    data['label'] = model.labels_
    data.to_csv('models//dbscan_labels.csv')
    return data,unlabelled_data,labelled_data
    

DATA_FOLDER = '..//uploads'
data,unlabelled,labelled = train(DATA_FOLDER+'//shady_customer_data.csv',eps=0.30)

print('\n')
# print(data)