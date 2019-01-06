#Package importing

import pandas as pd
import numpy as np
import matplotlib as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score,silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


original_data = pd.read_csv('data/tp2_data.csv').drop('Unnamed: 0',axis = 1)
original_data = original_data[original_data.type != 'nuclear explosion']

info = {'x':6371 * np.cos(original_data['latitude'].values * np.pi/180) * np.cos(original_data['longitude'].values * np.pi/180) , 'y': 6371 * np.cos(original_data['latitude'].values *np.pi/180) * np.sin(original_data['longitude'].values * np.pi/180) , 'z': 6371 * np.sin(original_data['latitude'].values * np.pi/180), 'fault': original_data['fault'], 'latitude': original_data['latitude'], 'longitude': original_data['longitude'] }

data = pd.DataFrame(data=info,columns=['x','y','z','fault','latitude','longitude'])


# Utility functions

def plot_classes(labels,lon,lat, alpha=0.5, edge = 'k'):

    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""

    img = plt.imread("Mollweide_projection_SW.jpg")

    plt.figure(figsize = (10,5),frameon = False)
    x = lon/180*np.pi
    y = lat/180*np.pi
    ax = plt.subplot(111, projection = "mollweide")
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x,y)).T)
    print(np.min(np.vstack((x,y)).T,axis = 0))
    print(np.min(t,axis = 0))
    clims = np.array([(-np.pi,0),(np.pi,0),(0,-np.pi/2),(0,np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize = (10,5),frameon = False)
    plt.subplot(111)
    plt.imshow(img,zorder = 0,extent = [lims[0,0],lims[1,0],lims[2,1],lims[3,1]],aspect = 1)
    x = t[:,0]
    y = t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)
    ix = 0
    for lab in diffs[diffs >= 0]:
        mask = labels == lab
        nots = np.logical_or(nots,mask)
        plt.plot(x[mask], y[mask],'o', markersize = 4, mew = 1,zorder = 1,alpha = alpha, markeredgecolor = edge)
        ix = ix+1
    mask = np.logical_not(nots)
    if np.sum(mask) > 0:
        plt.plot(x[mask], y[mask], '.', markersize = 1, mew = 1,markerfacecolor = 'w', markeredgecolor = edge)


#Function that calculates the metrics 
def rand_index(df, labels, groups):
    TP = TN = FP = FN = 0

    for i in range(len(labels)):
        SL = labels[i] == labels[(i+1):]
        SG = groups[i] == groups[(i+1):]

        TP_aux = np.logical_and(SG,SL)
        FP_aux = np.logical_and(np.logical_not(SG), SL)
        FN_aux = np.logical_and(SG, np.logical_not(SL))
        TN_aux = np.logical_and(np.logical_not(SG), np.logical_not(SL))

        TP += np.sum(TP_aux)
        TN += np.sum(TN_aux)
        FP += np.sum(FP_aux)
        FN += np.sum(FN_aux)

        Rand_score = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)

        return([Rand_score, Precision, Recall, F1, silhouette_score(df.loc[:,['x','y','z']],labels), adjusted_rand_score(labels,groups)])

#K-Means----
        
def get_plot_metrics_Kmeans(df, cluster_range, plot_metric = False, plot_clusters = False, get = False):

    label_array = np.zeros((df.shape[0],1 + max(cluster_range)))

    for n_clust in cluster_range:
        kmeans_labels = KMeans(n_clusters = n_clust).fit_predict(df.loc[:,['x','y','z']])
        label_array[:,n_clust] = kmeans_labels

    if plot_metric or get :
        metrics_df = pd.DataFrame(index = [], columns = ['Rand Score', 'Precision', 'Recall', 'F1 score',
                                                         'Silhouette score', 'Adjusted Rand Score'])

        for n_clust in cluster_range:
            metrics_df.loc[n_clust] = rand_index(df, label_array[:,n_clust] , df["fault"])

        if get:
            print(metrics_df)
        if plot_metric:
            metrics_df.plot(subplots = True)

    if plot_clusters:
        for n_clust in cluster_range:
            plot_classes(label_array[:,n_clust], df.longitude, df.latitude)
  
#GMM----          
def get_plot_metrics_Gmix(df, cluster_range, plot_metric = False, plot_clusters = True, get = True):
    
    label_array = np.zeros((data.shape[0],1 + max(cluster_range)))
    
    for n_clust in cluster_range:
        Gmix_labels = GaussianMixture(n_components = n_clust).fit_predict(df.loc[:,['x','y','z']])
        label_array[:,n_clust] = Gmix_labels
                
    if plot_metric or get : 
        metrics_df = pd.DataFrame(index = [], columns = ['Rand Score', 'Precision', 'Recall', 'F1 score',
                                                         'Silhouette score', 'Adjusted Rand Score'])

        for n_clust in cluster_range:
            metrics_df.loc[n_clust] = rand_index(df, label_array[:,n_clust], df["fault"])
        
        if get:
            print(metrics_df)
    
        if plot_metric:
            metrics_df.plot(subplots = True)
        
    if plot_clusters:
        for n_clust in cluster_range:
            plot_classes(label_array[:,n_clust], df.longitude, df.latitude)   

#Function that calculates the multivariable correlation matrix
def make_corr(covs):

    idx = ["Cluster " + str(i) for i in range(1,len(covs)+1)]
    cols = ["Cluster " + str(i) for i in range(1,len(covs)+1)]

    df_corrs = pd.DataFrame(index=idx, columns=cols)

    for i in range(0,5):
        corr=[]
        for j in range(0,5):
            a = np.matmul(np.transpose(covs[i]),covs[j])
            b = np.matmul(np.transpose(covs[j]),covs[i])

            xx = np.matmul(np.transpose(covs[i]),covs[i])
            yy = np.matmul(np.transpose(covs[j]),covs[j])

            trace = np.trace(np.matmul(a,b))

            x = np.trace(np.matmul(np.transpose(xx),xx))
            y = np.trace(np.matmul(yy,yy))

            corr.append(trace/np.sqrt(x*y))

        df_corrs.iloc[i]=corr
    
    return df_corrs.astype("float")
    
 
#DBSCAN----
    

def get_plot_metrics_DB(df, eps_range, plot_metric = False, plot_clusters = False, get = True):
    
    label_array = np.zeros((len(eps_range), 1 + df.shape[0]))
    label_array[:,0] = eps_range
    
    for count in range(len(eps_range)):
        
        DB_labels = DBSCAN(eps = eps_range[count]).fit_predict(df.loc[:,['x','y','z']])
        label_array[count ,1:] = DB_labels
                    
    if plot_metric or get : 
        
        metrics_df = pd.DataFrame(index = eps_range ,columns = ['Rand Score', 'Precision', 'Recall', 'F1 score',
                                                                'Silhouette score', 'Adjusted Rand Score'])
    
        for line in range(metrics_df.shape[0]):
            metrics_df.iloc[line] = rand_index(df, label_array[line,1:], df["fault"])
        
        if get:
            print(metrics_df)
    
        if plot_metric:
            metrics_df.plot(subplots = True)
        
    if plot_clusters:
        for count in label_array[:,1]:
            plot_classes(label_array[count,2:], df.longitude, df.latitude)    
    


#Data loading
original_data = pd.read_csv('data/tp2_data.csv').drop('Unnamed: 0',axis = 1)
original_data = original_data[original_data.type != 'nuclear explosion']

info = {'x':6371 * np.cos(original_data['latitude'].values * np.pi/180) * np.cos(original_data['longitude'].values * np.pi/180) , 'y': 6371 * np.cos(original_data['latitude'].values *np.pi/180) * np.sin(original_data['longitude'].values * np.pi/180) , 'z': 6371 * np.sin(original_data['latitude'].values * np.pi/180), 'fault': original_data['fault'], 'latitude': original_data['latitude'], 'longitude': original_data['longitude'] }

data = pd.DataFrame(data=info,columns=['x','y','z','fault','latitude','longitude'])

#Kmeans -----
get_plot_metrics_Kmeans(data,range(3,29), plot_metric = True, get = False)


#GMM -----
get_plot_metrics_Gmix(data, range(3,50), plot_metric = True, get = True)

#This calculates the multivariable correlation matrix and plots a heat map
#It is also worth adding that it was an attempt to find if clusters A and B that had high
#correlation meant that if A had a cluster, B also should have one. 
Gmix_labels = GaussianMixture(n_components = 5, random_state=4).fit(data.loc[:,['x','y','z']])
covs = Gmix_labels.covariances_
df_corrs = make_corr(covs)

sns.heatmap(df_corrs.astype(float), annot=True);

#DBSCAN -----

# Epsilon extraction for DBSCAN
Knn_model = KNeighborsClassifier(n_neighbors = 4).fit(data.loc[:,['x','y','z']], np.zeros(len(data)))
dists = np.sort(Knn_model.kneighbors()[0][:,3])[::-1]

#This is just to check the elbow plot, you can comment the code below in case it fails

py.sign_in('Nico_BigD','cUX0PdbJrgOmZeTKkoR1')

x_data = np.linspace(0,len(data),num = len(data) + 1)
y_data = dists

trace = go.Scatter(
    x = x_data,
    y = y_data,
    mode = 'markers'
)

data_plot = [trace]
py.iplot(data_plot, filename='Bazofe')

get_plot_metrics_DB(data,np.linspace(300, 500, num = 200),plot_metric = True, get = False)

