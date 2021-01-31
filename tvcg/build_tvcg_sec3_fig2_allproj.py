# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 09:09:50 2020

@author: Bruno Schneider, University of Konstanz
"""

import pandas as pd
import numpy as np
#from sklearn import preprocessing
#from lib.psi import calculate_psi
import app.dshift_features2 as dsfeat
import app.model_training as mt


original_data = pd.read_csv("experiments/tvcg/prototype-dev-3classes-no_drop.csv")
#original_data = pd.read_csv("experiments/tvcg/prototype-dev-3classes-drop.csv")
data_only_features_n_label = original_data[['f0', 'f1','f2', 'f3','f4', 'f5','f6', 'f7','label','subset']]
features_colnames = ['f0', 'f1','f2', 'f3','f4', 'f5','f6', 'f7']
training = data_only_features_n_label[data_only_features_n_label['subset'] == 'training']
unseen = data_only_features_n_label[data_only_features_n_label['subset'] == 'unseen']
num_features = 8
num_classes = 3
training_size = 2000
unseen_size = 1000
radius_search_size = 3
measures = dsfeat.DshiftFeatures2()

# ks
models = mt.ModelTraining()
ks_stats = models.compute_ks_statistics(training, unseen, num_features)

# projections  **** CHANGE HERE *******
#linear_proj = measures.umap_projetion_centroids(training, original_data, num_features, features_colnames, num_classes)
linear_proj = measures.mds_projetion_centroids(training, original_data, num_features, features_colnames, num_classes)
#linear_proj = measures.tsne_projetion_centroids(training, original_data, num_features, features_colnames, num_classes)

# train centroids
pca_train_class_centroids = linear_proj[0]
pca_df = measures.build_centroids_df(pca_train_class_centroids, "x1_pca", "y1_pca")


# train + unseen centroids
unseen_closest_centroids = measures.closest_centroid_foreach_newpoint(training.iloc[ : , 0:num_features+1 ], 
                                                                      unseen.iloc[ : , 0:num_features ])
unseen['label'] = unseen_closest_centroids
all_data_to_concat = [training, unseen]
all_data = pd.concat(all_data_to_concat)

alldata_proj = measures.kpca_projetion_centroids(all_data, original_data, num_features, features_colnames, num_classes)
pca_alldata_class_centroids = alldata_proj[0]
pca_unseen_df = measures.build_centroids_df(pca_alldata_class_centroids, "x2_pca", "y2_pca")

centroids_to_concat = [pca_df, pca_unseen_df]
centroids = pd.concat(centroids_to_concat, sort=False, axis=1)

# compute distances between training and unseen centroids
training_dspace_centroids = linear_proj[2].values
unseen_dspace_centroids = alldata_proj[2].values
distances = measures.compute_normalized_euclideandistances(training_dspace_centroids, unseen_dspace_centroids)


distances_change = measures.compute_distances_change_new(training_dspace_centroids, unseen_dspace_centroids)

centroids['distances'] = distances 

# prepare dataframe and generate columns for all features
unseen['label'] = np.nan # after computing the centroids, correct this column

    # density_in_train
#tree = measures.create_balltree(training.iloc[:,0:num_features], 40) # leaf_size default = 40
#density_in_train = measures.compute_density_scores_3classes(training, unseen.iloc[:,0:num_features], pd.DataFrame(training.label), tree,radius_search_size)
#training_density_in_train = measures.compute_density_scores_3classes(training, training.iloc[:,0:num_features], pd.DataFrame(training.label), tree,radius_search_size)
#
#    # density in unseen
#unseen_tree = measures.create_balltree(unseen.iloc[:,0:num_features], 40)
#density_in_unseen = measures.compute_density_scores_in_newdata(unseen.iloc[:,0:num_features], unseen_tree, radius_search_size)
#
#    # density_diff
#density_diff = np.array(density_in_unseen) - np.array(density_in_train)  
#
#training['r'] = training_density_in_train
#training['r_diff'] = np.nan
#unseen['r'] = density_in_train
#unseen['r_diff'] = density_diff
#
unseen['closest_centroid'] = unseen_closest_centroids
train_closest_centroids = measures.closest_centroid_foreach_newpoint(training.iloc[ : , 0:num_features+1 ], 
                                                                      training.iloc[ : , 0:num_features ])
training['closest_centroid'] = train_closest_centroids

# CONCATENATING TRAIN AND UNSEEN
all_data = pd.concat([training, unseen], sort=False) 

x_pca = linear_proj[1][:,0]
y_pca = linear_proj[1][:,1]
all_data['x_pca'] = x_pca
all_data['y_pca'] = y_pca


all_data['x_mds'] = np.nan
all_data['y_mds'] = np.nan

# recuperando as colunas r e r_diff
all_data['r'] = np.nan
all_data['r'] = original_data['r'] 

all_data['r_diff'] = np.nan
all_data['r_diff'] = original_data['r_diff'] 

## write csvs
#centroids.to_csv('experiments/tvcg/tvcg_centroids_nodrop_sec3_fig2_mds.csv', index = False)
#all_data.to_csv('experiments/tvcg/tvcg_data_nodrop_sec3_fig2_mds.csv', index = False)


 

