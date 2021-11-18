import numpy as np
from numpy.linalg import eig
from sklearn import decomposition

#load dataset from .csv file
seeds = np.loadtxt('seeds.csv', skiprows=1, delimiter=',')

#delete type label
seeds_without_type = np.delete(seeds, 7, axis=1)

#calculate centering and means
mean_value = np.mean(seeds_without_type, axis=0)
centering = seeds_without_type-mean_value
centering_transpose = np.transpose(centering)

#covariance matrix
covariance = np.cov(centering_transpose)
covariance_transpose = np.transpose(covariance)

#eigen values/vectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_transpose)
sorting = np.argsort(eigen_values)[::-1]

#explained variance and ratio
explained_variance = eigen_values[sorting]
explained_variance_ratio = explained_variance/np.sum(explained_variance)
print("EXPLAINED WITH NATIVE LIBRARIES")
print(explained_variance)
print(explained_variance_ratio)

#using library pca
pca = decomposition.PCA(7)
X_train_pca = pca.fit_transform(seeds_without_type)
explained_variance_pca = pca.explained_variance_
explained_variance_ratio_pca = pca.explained_variance_ratio_
print("EXPLAINED WITH PCA")
print(explained_variance_pca)
print(explained_variance_ratio_pca)

print("COMPARISON RESULTS:")
print(np.sum(explained_variance_ratio_pca))
print(np.sum(explained_variance_ratio))
