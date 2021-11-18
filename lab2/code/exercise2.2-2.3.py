import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

#plots and coloring based on types
def plots(types, x, y, which_algorithm_text, which_algorithm_number):
    if which_algorithm_number == 1:
        colors_1 = []
        for type in types:
            if type == -1:
                colors_1.append("black")
            elif type == 1:
                colors_1.append("yellow")
            elif type == 2:
                colors_1.append("blue")
            elif type == 3:
                colors_1.append("red")
            elif type == 4:
                colors_1.append("green")
            elif type == 5:
                colors_1.append("grey")
            elif type == 6:
                colors_1.append("teal")
            elif type == 7:
                colors_1.append("brown")
            elif type == 8:
                colors_1.append("orange")
            elif type == 9:
                colors_1.append("purple")
            elif type == 10:
                colors_1.append("pink")
            elif type == 11:
                colors_1.append("olive")
            elif type == 12:
                colors_1.append("navy")
            elif type == 13:
                colors_1.append("cyan")
            elif type == 14:
                colors_1.append("gold")
            else:
                colors_1.append("magenta")
        plt.scatter(x, y, c=colors_1)
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Data Set 1(' + which_algorithm_text + ') - No. Cluster 15')
        type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
        type2_patch = mpatches.Patch(color='blue', label='Class of type 2')
        type3_patch = mpatches.Patch(color='red', label='Class of type 3')
        type4_patch = mpatches.Patch(color='green', label='Class of type 4')
        type5_patch = mpatches.Patch(color='grey', label='Class of type 5')
        type6_patch = mpatches.Patch(color='teal', label='Class of type 6')
        type7_patch = mpatches.Patch(color='brown', label='Class of type 7')
        type8_patch = mpatches.Patch(color='orange', label='Class of type 8')
        type9_patch = mpatches.Patch(color='purple', label='Class of type 9')
        type10_patch = mpatches.Patch(color='pink', label='Class of type 10')
        type11_patch = mpatches.Patch(color='olive', label='Class of type 11')
        type12_patch = mpatches.Patch(color='navy', label='Class of type 12')
        type13_patch = mpatches.Patch(color='cyan', label='Class of type 13')
        type14_patch = mpatches.Patch(color='gold', label='Class of type 14')
        type15_patch = mpatches.Patch(color='magenta', label='Class of type 15')
        type16_patch = mpatches.Patch(color='black', label='Class noise')
        plt.legend(
            handles=[type1_patch, type2_patch, type3_patch, type4_patch, type5_patch, type6_patch, type7_patch,
                     type8_patch,
                     type9_patch, type10_patch, type11_patch, type12_patch, type13_patch, type14_patch,
                     type15_patch, type16_patch])
        plt.show()
    elif which_algorithm_number == 2:
        colors_2 = []
        for type in types:
            if type == 1:
                colors_2.append("yellow")
            elif type == -1:
                colors_2.append("black")
            else:
                colors_2.append("blue")
        plt.scatter(x, y, c=colors_2)
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Data Set 2(' + which_algorithm_text + ') - No. Cluster 2')
        type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
        type2_patch = mpatches.Patch(color='blue', label='Class of type 2')
        type3_patch = mpatches.Patch(color='black', label='Class noise')
        plt.legend(handles=[type1_patch, type2_patch,type3_patch])
        plt.show()
    elif which_algorithm_number == 3:
        colors_3 = []
        for type in types:
            if type == 1:
                colors_3.append("yellow")
            elif type == -1:
                colors_3.append("black")
            else:
                colors_3.append("blue")
        plt.scatter(x, y, c=colors_3)
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Data Set 3(' + which_algorithm_text + ') - No. Cluster 2')
        type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
        type2_patch = mpatches.Patch(color='blue', label='Class of type 2')
        type3_patch = mpatches.Patch(color='black', label='Class noise')
        plt.legend(handles=[type1_patch, type2_patch, type3_patch])
        plt.show()
    else:
        colors_4 = []
        for type in types:
            if type == 1:
                colors_4.append("yellow")
            elif type == -1:
                colors_4.append("black")
            else:
                colors_4.append("blue")
        plt.scatter(x, y, c=colors_4)
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Data Set 4(' + which_algorithm_text + ') - No. Cluster 2')
        type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
        type2_patch = mpatches.Patch(color='blue', label='Class of type 2')
        type3_patch = mpatches.Patch(color='black', label='Class noise')
        plt.legend(handles=[type1_patch, type2_patch, type3_patch])
        plt.show()


#import data from datasets
dataset1 = np.loadtxt('dataset1_noCluster15.csv', skiprows=1, delimiter=',')
dataset2 = np.loadtxt('dataset2_noCluster2.csv', skiprows=1, delimiter=',')
dataset3 = np.loadtxt('dataset3_noCluster2.csv', skiprows=1, delimiter=',')
dataset4 = np.loadtxt('dataset4_noCluster2.csv', skiprows=1, delimiter=',')

#extracting type labels from datasets
types_1 = dataset1[:,2]
types_2 = dataset2[:,2]
types_3 = dataset3[:,2]
types_4 = dataset4[:,2]

#deleting type column from datasets
np.delete(dataset1, 2, axis=1)
np.delete(dataset2, 2, axis=1)
np.delete(dataset3, 2, axis=1)
np.delete(dataset4, 2, axis=1)

#dataprocessing
processing_scalar = StandardScaler()
processing_1 = processing_scalar.fit_transform(dataset1)
processing_2 = processing_scalar.fit_transform(dataset2)
processing_3 = processing_scalar.fit_transform(dataset3)
processing_4 = processing_scalar.fit_transform(dataset4)

#algorithms for clustering

#dbscan algorithm + normalization + rand index with labels for each dataset
dbscan_algorithm_1 = DBSCAN(eps=0.125, min_samples=4).fit(processing_1)
labels_1 = dbscan_algorithm_1.labels_
print('Normalized Mutual Information DBSCAN DataSet1: ',normalized_mutual_info_score(types_1, labels_1))
print('Adjusted Rand Score DBSCAN Dataset1: ',adjusted_rand_score(types_1, labels_1))
dbscan_algorithm_2 = DBSCAN(eps=0.5, min_samples=4).fit(processing_2)
labels_2 = dbscan_algorithm_2.labels_
print('Normalized Mutual Information DBSCAN DataSet2: ',normalized_mutual_info_score(types_2, labels_2))
print('Adjusted Rand Score DBSCAN Dataset2: ',adjusted_rand_score(types_2, labels_2))
dbscan_algorithm_3 = DBSCAN(eps=0.47, min_samples=4).fit(processing_3)
labels_3 = dbscan_algorithm_3.labels_
normalized_mutual_info_score(types_3, labels_3)
adjusted_rand_score(types_3, labels_3)
print('Normalized Mutual Information DBSCAN DataSet3: ',normalized_mutual_info_score(types_3, labels_3))
print('Adjusted Rand Score DBSCAN Dataset3: ',adjusted_rand_score(types_3, labels_3))
dbscan_algorithm_4 = DBSCAN(eps=0.35, min_samples=4).fit(processing_4)
labels_4 = dbscan_algorithm_4.labels_
normalized_mutual_info_score(types_4, labels_4)
adjusted_rand_score(types_4, labels_4)
print("Normalized Mutual Information DBSCAN DataSet4:",normalized_mutual_info_score(types_4, labels_4))
print('Adjusted Rand Score DBSCAN Dataset4: ', adjusted_rand_score(types_4, labels_4))

# kmeans algorithm + normalization + rand index with labels for each dataset
kmeans_algorithm_1 = KMeans(n_clusters=15, random_state=1).fit(processing_1)
labels_km_1 = kmeans_algorithm_1.labels_
print('Normalized Mutual Information KMeans DataSet1: ',normalized_mutual_info_score(types_1, labels_km_1))
print('Adjusted Rand Score KMeans Dataset1: ',adjusted_rand_score(types_1, labels_km_1))
kmeans_algorithm_2 = KMeans(n_clusters=2, random_state=1).fit(processing_2)
labels_km_2 = kmeans_algorithm_2.labels_
print('Normalized Mutual Information KMeans DataSet2: ',normalized_mutual_info_score(types_2, labels_km_2))
print('Adjusted Rand Score KMeans Dataset2: ',adjusted_rand_score(types_2, labels_km_2))
kmeans_algorithm_3 = KMeans(n_clusters=2, random_state=1).fit(processing_3)
labels_km_3 = kmeans_algorithm_3.labels_
print('Normalized Mutual Information KMeans DataSet3: ',normalized_mutual_info_score(types_3, labels_km_3))
print('Adjusted Rand Score KMeans Dataset3: ',adjusted_rand_score(types_3, labels_km_3))
kmeans_algorithm_4 = KMeans(n_clusters=2, random_state=1).fit(processing_4)
labels_km_4 = kmeans_algorithm_4.labels_
print('Normalized Mutual Information KMeans DataSet4: ',normalized_mutual_info_score(types_4, labels_km_4))
print('Adjusted Rand Score KMeans Dataset4: ',adjusted_rand_score(types_4, labels_km_4))

# expectation maximization algorithm + normalization + rand index with labels for each dataset
expectation_maximization_algorithm_1 = GaussianMixture(n_components=15, random_state=0).fit(processing_1)
labels_ex_1 = expectation_maximization_algorithm_1.fit(processing_1).predict(processing_1)
print('Normalized Mutual Information Expectation Maximization DataSet1: ',normalized_mutual_info_score(types_1, labels_ex_1))
print('Adjusted Rand Score Expectation Maximization Dataset1: ',adjusted_rand_score(types_1, labels_ex_1))
expectation_maximization_algorithm_2 = GaussianMixture(n_components=2, random_state=0).fit(processing_2)
labels_ex_2 = expectation_maximization_algorithm_2.fit(processing_2).predict(processing_2)
print('Normalized Mutual Information Expectation Maximization DataSet2: ',normalized_mutual_info_score(types_2, labels_ex_2))
print('Adjusted Rand Score Expectation Maximization Dataset2: ',adjusted_rand_score(types_2, labels_ex_2))
expectation_maximization_algorithm_3 = GaussianMixture(n_components=2, random_state=0).fit(processing_3)
labels_ex_3 = expectation_maximization_algorithm_3.fit(processing_3).predict(processing_3)
print('Normalized Mutual Information Expectation Maximization DataSet3: ',normalized_mutual_info_score(types_3, labels_ex_3))
print('Adjusted Rand Score Expectation Maximization Dataset3: ',adjusted_rand_score(types_3, labels_ex_3))
expectation_maximization_algorithm_4 = GaussianMixture(n_components=2, random_state=0).fit(processing_4)
labels_ex_4 = expectation_maximization_algorithm_4.fit(processing_4).predict(processing_4)
print('Normalized Mutual Information Expectation Maximization DataSet4: ',normalized_mutual_info_score(types_4, labels_ex_4))
print('Adjusted Rand Score Expectation Maximization Dataset4: ',adjusted_rand_score(types_4, labels_ex_4))

# average link algorithm + normalization + rand index with labels for each dataset
average_link_algorithm_1 = AgglomerativeClustering(n_clusters=15).fit(processing_1)
labels_al_1 = average_link_algorithm_1.labels_
print('Normalized Mutual Information Average Link DataSet1: ',normalized_mutual_info_score(types_1, labels_al_1))
print('Adjusted Rand Score Expectation Average Link Dataset1: ',adjusted_rand_score(types_1, labels_al_1))
average_link_algorithm_2 = AgglomerativeClustering(n_clusters=2).fit(processing_2)
labels_al_2 = average_link_algorithm_2.labels_
print('Normalized Mutual Information Average Link DataSet2: ',normalized_mutual_info_score(types_2, labels_al_2))
print('Adjusted Rand Score Expectation Average Link Dataset2: ',adjusted_rand_score(types_2, labels_al_2))
average_link_algorithm_3 = AgglomerativeClustering(n_clusters=2).fit(processing_3)
labels_al_3 = average_link_algorithm_3.labels_
print('Normalized Mutual Information Average Link DataSet3: ',normalized_mutual_info_score(types_3, labels_al_3))
print('Adjusted Rand Score Expectation Average Link Dataset3: ',adjusted_rand_score(types_3, labels_al_3))
average_link_algorithm_4 = AgglomerativeClustering(n_clusters=2).fit(processing_4)
labels_al_4 = average_link_algorithm_4.labels_
print('Normalized Mutual Information Average Link DataSet4: ',normalized_mutual_info_score(types_4, labels_al_4))
print('Adjusted Rand Score Expectation Average Link Dataset4: ',adjusted_rand_score(types_4, labels_al_4))

# plot dataset1
x_1 = processing_1[:, 0]
y_1 = processing_1[:, 1]
plots(types_1, x_1, y_1, '', 1)
plots(labels_1, x_1, y_1, 'DBSCAN Algorithm', 1)
plots(labels_km_1, x_1, y_1, 'KMeans Algorithm', 1)
plots(labels_ex_1, x_1, y_1, 'Expectation Maximization Algorithm', 1)
plots(labels_al_1, x_1, y_1, 'Average Link Algorithm', 1)

# plot dataset2
x_2 = processing_2[:, 0]
y_2 = processing_2[:, 1]
plots(types_2, x_2, y_2, '', 2)
plots(labels_2, x_2, y_2, 'DBSCAN Algorithm', 2)
plots(labels_km_2, x_2, y_2, 'KMeans Algorithm', 2)
plots(labels_ex_2, x_2, y_2, 'Expectation Maximization Algorithm', 2)
plots(labels_al_2, x_2, y_2, 'Average Link Algorithm', 2)

# plot dataset3
x_3 = processing_3[:, 0]
y_3 = processing_3[:, 1]
plots(types_3, x_3, y_3, '', 3)
plots(labels_3, x_3, y_3, 'DBSCAN Algorithm', 3)
plots(labels_km_3, x_3, y_3, 'KMeans Algorithm', 3)
plots(labels_ex_3, x_3, y_3, 'Expectation Maximization Algorithm', 3)
plots(labels_al_3, x_3, y_3, 'Average Link Algorithm', 3)

# plot dataset4
x_4 = processing_4[:, 0]
y_4 = processing_4[:, 1]
plots(types_4, x_4, y_4, '', 4)
plots(labels_4, x_4, y_4, 'DBSCAN Algorithm', 4)
plots(labels_km_4, x_4, y_4, 'KMeans Algorithm', 4)
plots(labels_ex_4, x_4, y_4, 'Expectation Maximization Algorithm', 4)
plots(labels_al_4, x_4, y_4, 'Average Link Algorithm', 4)

