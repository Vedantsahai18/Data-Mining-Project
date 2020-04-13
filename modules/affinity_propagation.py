# Affinity Propagation clustering
from numpy import unique
from numpy import where
import numpy as np
import pickle

from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

import pandas as pd
from matplotlib import pyplot as plt

def preprocessor(df):
    df = df.as_matrix().astype('float32',copy=False)
    #Small preprocessing Pipeline
    stscaler = StandardScaler().fit(df)
    df = stscaler.transform(df)
    return df
    
def train(dataset):
    data = pd.read_csv(dataset)
    data.fillna(method = 'ffill',inplace=True)
    IMAGE_FOLDER = '..//images//'
    MODEL_FOLDER = 'models//'

    header = data.columns.values
    plt.figure(0)
    
    
    plt.scatter(data[header[0]],data[header[1]])
    unlabelled_data = IMAGE_FOLDER+'affinity_propagation_unlabelled.png'
    plt.savefig(unlabelled_data)

    df = preprocessor(data)

    model = AffinityPropagation(damping=0.9)

    # filename = MODEL_FOLDER  +'dbscan.sav'

    # pickle.dump(model, open(filename, 'wb'))
    model.fit(df)
    yhat = model.predict(df)
    # retrieve unique clusters
    clusters = unique(yhat)

    plt.figure(1)

    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(df[row_ix, 0], df[row_ix, 1])
    labelled_data = IMAGE_FOLDER+'affinity_propagation'+'.png'
    plt.savefig(labelled_data)
    # show the plot

    data['label'] = model.labels_
    data.to_csv('models//affinity_propagation_labels.csv')
    return data,unlabelled_data,labelled_data
    

DATA_FOLDER = '..//uploads'
data,unlabelled,labelled = train(DATA_FOLDER+'//shady_customer_data.csv')

print('\n')
print(data)