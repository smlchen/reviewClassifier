#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ### Read the data

# In[2]:


# Read tfidf matrices.
test = pd.read_csv("~/Documents/ECS171/split/1/test_data.csv")
train = pd.read_csv("~/Documents/ECS171/split/1/training_data.csv")

# Combine test and training set for tfidf.
combined_tfidf = pd.concat([test, train])

# Read doc2vec data.
combined_d2v = pd.read_csv("~/Documents/ECS171/doc2vec.csv")


# In[101]:


combined_tfidf.head()


# In[3]:


# Convert Vector col from str to vector.
combined_d2v['Vector'] = [np.array(np.matrix(combined_d2v['Vector'][i])).ravel() 
                          for i in range(len(combined_d2v['Vector']))]

# Separate doc2vec vector into columns.
split = pd.DataFrame(combined_d2v['Vector'].values.tolist())
combined_d2v = pd.concat([combined_d2v.iloc[:, 0:2], split], axis=1)


# The star ratings and tfidf/doc2vec matrix is in the same dataframe. We have to separate the star ratings from the matrices in order to perform clustering.

# In[118]:


# Unbalanced classes.

# Separate vector and categories/ratings.
vec_tfidf = combined_tfidf.loc[:, ~combined_tfidf.columns.isin(['output1', 'output2'])].copy()
outputs_tfidf = combined_tfidf.loc[:, combined_tfidf.columns.isin(['output1', 'output2'])].copy()

# Separate vector and categories/ratings.
vec_d2v = combined_d2v.loc[:, ~combined_d2v.columns.isin(['cleanText', 'overall'])].copy()
outputs_d2v = combined_d2v.loc[:, combined_d2v.columns.isin(['cleanText', 'overall'])].copy()

# Rename the column with the star rating so that they have the same name.
outputs_tfidf = outputs_tfidf.rename(columns = {'output2' : 'rating'})
outputs_tfidf.head()

outputs_d2v = outputs_d2v.rename(columns = {'overall' : 'rating'})
outputs_d2v.head()


# In[ ]:


# Fix class imbalance by downsampling.
tmp_tfidf = combined_tfidf.groupby(['output2'], group_keys=False)
balanced_df_tfidf = pd.DataFrame(tmp_tfidf.apply(lambda x: x.sample(tmp_tfidf.size().min()))).reset_index(drop=True)

tmp_d2v = combined_d2v.groupby(['overall'], group_keys=False)
balanced_df_d2v = pd.DataFrame(tmp_d2v.apply(lambda x: x.sample(tmp_d2v.size().min()))).reset_index(drop=True)

# Separate vector and categories/ratings. Scale the vectors to get mean 0 and var 1.
vec_tfidf = balanced_df_tfidf.loc[:, ~balanced_df_tfidf.columns.isin(['output1', 'output2'])].copy()
outputs_tfidf = balanced_df_tfidf.loc[:, balanced_df_tfidf.columns.isin(['output1', 'output2'])].copy()

# Separate vector and categories/ratings. Scale the vectors to get mean 0 and var 1.
vec_d2v = balanced_df_d2v.loc[:, ~balanced_df_d2v.columns.isin(['cleanText', 'overall'])].copy()
outputs_d2v = balanced_df_d2v.loc[:, balanced_df_d2v.columns.isin(['cleanText', 'overall'])].copy()

# rename column
outputs_tfidf = outputs_tfidf.rename(columns = {'output2' : 'rating'})
outputs_tfidf.head()

outputs_d2v = outputs_d2v.rename(columns = {'overall' : 'rating'})
outputs_d2v.head()


# ### Generate k-mean groups

# In[106]:


def perform_kmeans(df_labels, df_vec, num_clusters):
    """
    Input:
        df_labels -- dataframe with the true labels
        df_vec -- feature matrix
        num_clusters -- number of clusters
    Output: 
        df_labels with an additional column of k-means clusters labels
        
    This function performs k-means clustering on df_vec.
    """
    # K-means clustering with 5 clusters
    km = KMeans(n_clusters = num_clusters)
    km.fit(df_vec)
    clusters = km.labels_.tolist()
    
    # Add cluster labels to dataframe
    df_labels['cluster'] = clusters
    
    return df_labels


# In[22]:


# # K-means results with tfidf
# df_tfidf_kmeans = perform_kmeans(outputs_tfidf, vec_tfidf, 5)


# In[27]:


# # K-means results with doc2vec
# df_d2v_kmeans = perform_kmeans(outputs_d2v, vec_d2v, 5)


# In[113]:


# Pickle kmeans dataframes so we don't have to re run kmeans.
# df_tfidf_kmeans.to_pickle("./kmeans/df_tfidf_kmeans")
# df_d2v_kmeans.to_pickle("./kmeans/df_d2v_kmeans")

df_tfidf_kmeans = pd.read_pickle("./clustering_results/df_tfidf_kmeans")
df_d2v_kmeans = pd.read_pickle("./clustering_results/df_d2v_kmeans")


# ### Dimensionality reduction with PCA

# In[5]:


def perform_pca(df_labels, df_vec):
    """
    Input:
        df_labels -- dataframe with the true labels
        df_vec -- feature matrix
    Output: 
        df with clusters, pc1, pc2
        
    This function reduces the dimensionality of the feature matrix to 2 dimensions using PCA.
    """
    pca = PCA(n_components = 2)
    pca_result = pca.fit_transform(df_vec)
    pc1, pc2 = pca_result[:, 0], pca_result[:, 1]
    
    # Create df with labels, clusters, and principle components
    df = pd.DataFrame(dict(rating=df_labels['rating'], cluster=df_labels['cluster'], pc1=pc1, pc2=pc2))
    
    return df


# In[122]:


# Perform pca and save the result as a pickle file.

# pca_tfidf = perform_pca(df_tfidf_kmeans, vec_tfidf)
# pca_tfidf.to_pickle("./clustering_results/pca_tfidf")
pca_tfidf = pd.read_pickle("./clustering_results/pca_tfidf")


# In[124]:


# Plot tfidf PCA results.
plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='pc1', y='pc2', data=pca_tfidf, hue='cluster', palette=sns.color_palette("hls", 5), 
                     style = 'rating', s = 100, alpha = 0.3)


# In[123]:


# Perform pca and save the result as a pickle file.
# pca_d2v = perform_pca(df_d2v_kmeans, vec_d2v)
# pca_d2v.to_pickle("./clustering_results/pca_d2v")
pca_tfidf = pd.read_pickle("./clustering_results/pca_tfidf")


# In[125]:


# Plot doc2vec PCA results.
plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='pc1', y='pc2', data=pca_d2v, hue='cluster', palette=sns.color_palette("hls", 5), 
                     style = 'rating', s = 100, alpha = 0.3)


# ### Dimensionality reduction with t-SNE

# In[103]:


def perform_tsne(df_labels, df_vec, lr):
    """
    Input:
        df_labels -- dataframe with the true labels
        tfidf -- tfidf matrix
        num_clusters -- number of clusters
    Output: 
        df with clusters, tsne1, tsne2
        
    This function reduces the dimensionality of the feature matrix to 2 dimensions using t-SNE.
    """
    tsne = TSNE(n_components=2, learning_rate = lr)
    tsne_results = tsne.fit_transform(df_vec)
    tsne1, tsne2 = tsne_results[:, 0], tsne_results[:, 1]
    
    # Create df with labels, clusters, and principle components
    df = pd.DataFrame(dict(rating=df_labels['rating'], cluster=df_labels['cluster'], tsne1=tsne1, tsne2=tsne2))
    
    return df


# In[11]:


# Perform k means on the balanced data.

# df_tfidf_kmeans = perform_kmeans(outputs_tfidf, vec_tfidf, 5)
# df_d2v_kmeans = perform_kmeans(outputs_d2v, vec_d2v, 5)

# df_tfidf_kmeans.to_pickle("./clustering_results/df_tfidf_tsne_balanced")
# df_d2v_kmeans.to_pickle("./clustering_results/df_d2v_tsne_balanced")
df_tfidf_tsne_balanced = pd.read_pickle("./clustering_results/df_tfidf_tsne_balanced")
df_d2v_tsne_balanced = pd.read_pickle("./clustering_results/df_d2v_tsne_balanced")


# In[29]:


# Perform t-SNE on the tfidf matrix.

#df_tfidf_tsne = perform_tsne(df_tfidf_kmeans, vec_tfidf)
#df_tfidf_tsne.to_pickle("./clustering_results/df_tfidf_tsne")
df_tfidf_tsne = pd.read_pickle("./clustering_results/df_tfidf_tsne")


# In[71]:


# tfidf plot using t-SNE
plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='tsne1', y='tsne2', data=df_tfidf_tsne, hue='cluster', palette=sns.color_palette("hls", 5), 
                 style = 'rating', s = 100, alpha = 0.7)


# In[31]:


# Perform t-SNE on the doc2vec matrix.
#df_d2v_tsne = perform_tsne(df_d2v_kmeans, vec_d2v)
#df_d2v_tsne.to_pickle("./clustering_results/df_d2v_tsne")
df_d2v_tsne = pd.read_pickle("./clustering_results/df_d2v_tsne")


# In[74]:


# doc2vec plot using t-SNE
plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='tsne1', y='tsne2', data=df_d2v_tsne, hue='cluster', palette=sns.color_palette("hls", 5), 
                 style = 'rating', s = 100, alpha = 0.7)


# In[81]:


# Combine plots into one figure
fig = plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)
ax = sns.scatterplot(x='tsne1', y='tsne2', data=df_tfidf_tsne, hue='cluster', palette=sns.color_palette("hls", 5), 
                 style = 'rating', s = 50, alpha = 0.5)

plt.subplot(1, 2, 2)
ax = sns.scatterplot(x='tsne1', y='tsne2', data=df_d2v_tsne, hue='cluster', palette=sns.color_palette("hls", 5), 
                 style = 'rating', s = 50, alpha = 0.5)


# In[90]:


fig = plt.figure(figsize=(10,10))

plt.subplot(2, 1, 1)
ax = sns.scatterplot(x='tsne1', y='tsne2', data=df_tfidf_tsne, hue='cluster', palette=sns.color_palette("hls", 5), 
                 style = 'rating', s = 50, alpha = 0.5)

plt.subplot(2, 1, 2)
ax = sns.scatterplot(x='tsne1', y='tsne2', data=df_d2v_tsne, hue='cluster', palette=sns.color_palette("hls", 5), 
                 style = 'rating', s = 50, alpha = 0.5)


# In[ ]:




