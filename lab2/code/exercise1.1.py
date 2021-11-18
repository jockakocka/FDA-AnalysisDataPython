import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

#load .csv file
seeds = np.loadtxt('seeds.csv', skiprows=1, delimiter=',')

#calculate means and center
mean_value = np.mean(seeds, axis=0)
centering = seeds-mean_value
centering_transpose = np.transpose(centering)

#calculate covariance matrix
covariance = np.cov(centering_transpose)

#calculate eigen vectors and values
eigen_values, eigen_vectors = np.linalg.eig(covariance)

#sort eigen vectors/values
eigen_vectors_transpose = eigen_vectors.transpose()
sorting = np.argsort(eigen_values)[::-1]
eigen_values_sorted = eigen_values[sorting]
eigen_vectors_sorted = eigen_vectors_transpose[sorting]

#reduced data
eigen_reduced = eigen_vectors_sorted[0:8]

#project data
projected_data = np.dot(centering, eigen_reduced.T)
print(projected_data)

