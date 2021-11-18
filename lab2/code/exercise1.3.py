import numpy as np
from numpy.linalg import eig
from sklearn import decomposition
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#load dataset from .csv file and extract types
seeds = np.loadtxt('seeds.csv', skiprows=1, delimiter=',')
types = seeds[:,7]

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
eigen_vectors_transpose = eigen_vectors.transpose()
eigen_vectors_sorted = eigen_vectors_transpose[sorting]


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

#projected data
eigen_reduced = eigen_vectors_sorted[0:7]
eigen_reduced_highest = eigen_reduced[0:2]
eigen_reduced_lowest = eigen_reduced[5:7]
projected_data_highest = np.dot(centering, eigen_reduced_highest.T)
projected_data_lowest = np.dot(centering, eigen_reduced_lowest.T)
print(projected_data_highest)
print(projected_data_lowest)

#coloring
colors = []
for type in types:
    if type == 1:
        colors.append("yellow")
    elif type == 2:
        colors.append("red")
    else:
        colors.append("blue")

#plot highest variance
highest1 = projected_data_highest[:,0]
highest2 = projected_data_highest[:,1]
plt.scatter(highest1, highest2,c=colors)
plt.xlabel('PCA1 Value of Highest Variance')
plt.ylabel('PCA2 Value of Highest Variance')
plt.title('PCA Components Highest Variance Dimensions')
type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
type2_patch = mpatches.Patch(color='red', label='Class of type 2')
type3_patch = mpatches.Patch(color='blue', label='Class of type 3')
plt.legend(handles=[type1_patch,type2_patch,type3_patch])
plt.show()

#plot lowest variance
lowest1 = projected_data_lowest[:,0]
lowest2 = projected_data_lowest[:,1]
plt.scatter(lowest1, lowest2, c=colors)
plt.xlabel('PCA6 Value of Lowest Variance')
plt.ylabel('PCA7 Value of Lowest Variance')
plt.title('PCA Components Lowest Variance Dimensions')
type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
type2_patch = mpatches.Patch(color='red', label='Class of type 2')
type3_patch = mpatches.Patch(color='blue', label='Class of type 3')
plt.legend(handles=[type1_patch,type2_patch,type3_patch])
plt.show()
