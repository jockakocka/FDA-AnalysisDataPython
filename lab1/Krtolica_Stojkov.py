import matplotlib.pyplot as plt
import numpy as np
from kfda import Kfda

# In this function, the different learning curves based on the gamma value is
# done. The training curve is shown in blue while the test curve is shown in
# yellow.
def draw_learning_curve_different_gamma(train_data_f, train_data_l, gamma_value):
    training_samples = np.array([])
    train_scores = np.array([])
    test_scores = np.array([])
    train_means = np.array([])
    test_means = np.array([])
    cls1 = Kfda(kernel='rbf', n_components=1, gamma=gamma_value)
    for count in range(5, 526, 5):
        training_samples = np.append(training_samples, count)
        for i in range(20):
            if count != 525:
                temp = 525 - count
                n = np.random.randint(temp)
                while True:
                    # [n: n + count,:]
                    # print("n")
                    # print(n)
                    # print("count")
                    # print(count)
                    # print("n+count")
                    # print(n+count)
                    train_data_features_sublist = train_data_f[n:n + count, :]
                    train_data_labels_sublist = train_data_l[n: n + count]
                    if np.where(train_data_labels_sublist == 0)[0].size > 0 and \
                            np.where(train_data_labels_sublist == 1)[0].size > 0:
                        break
                    else:
                        n = np.random.randint(temp)
            else:
                train_data_features_sublist = train_data_features
                train_data_labels_sublist = train_data_labels
            train_cls = cls1.fit(train_data_features_sublist, train_data_labels_sublist)
            train_score = cls1.score(train_data_features_sublist, train_data_labels_sublist)
            test_score = cls1.score(test_data_features, test_data_labels)
            train_scores = np.append(train_scores, train_score)
            test_scores = np.append(test_scores, test_score)
        train_means = np.append(train_means, np.mean(train_scores))
        test_means = np.append(test_means, np.mean(test_scores))
        train_scores = np.array([])
        test_scores = np.array([])
    plt.plot(training_samples, train_means, label='Training score', color='blue')
    plt.plot(training_samples, test_means, label='Test score', color='yellow')
    plt.xlabel('Number of training samples')
    plt.ylabel('Training and test score')
    plt.title('Learning curve with gamma = ' + str(gamma_value))
    plt.legend()
    plt.show()
    transformation_features_curve = cls.transform(train_data_f)
    print(separation_ration(transformation_features_curve, train_data_l))

# In this function, the plotting of the histograms is done. We are
# transforming the features and then extract the features based on
# their labels. Additionally the ratio is received is being calculated
# for each gamma value. On the histogram the plotting is done in two colors
# different for each class.
def plot_histogram(all_features, all_labels, gamma_value):
    cls = Kfda(kernel='rbf', n_components=1, gamma=gamma_value)
    cls.fit(all_features, all_labels)

    transformation_features_inside = cls.transform(all_features)
    print(transformation_features_inside)

    label_one_list = []
    label_zero_list = []
    for i in range(700):
        if all_labels[i] == 1:
            label_one_list.append(transformation_features_inside[i, 0])
        else:
            label_zero_list.append(transformation_features_inside[i, 0])
    ratio = separation_ration(transformation_features_inside, labels)
    plt.hist(label_zero_list, bins=100, color="blue", label="Label 0")
    plt.hist(label_one_list, bins=100, color="yellow", label="Label 1")
    plt.legend()
    plt.title("Histogram for gamma=" + str(gamma_value) + ",R =" + str(ratio))
    plt.show()

# In this function, the ratio is being computed. First the
# features based on their labels are extracted. Then the means
# are calculated for each of the classes. The distance is the
# squared difference between the mean 0 and mean 1. Then the sigma
# is calculated or the covariance matrices of the classes. The overall
# sigma is the sum of the two covariance matrices. Finally, the ratio
# is computed as the division between the squared distance and the
# overall sigma.
def separation_ration(projected_data, labels):
    i0 = np.where(labels == 0)[0]
    i1 = np.where(labels != 0)[0]
    print('----------')
    print(i0)

    u0 = np.mean(projected_data[i0, :], axis=0)
    u1 = np.mean(projected_data[i1, :], axis=0)
    distance = (u0 - u1)
    distance_squared = distance * distance

    sigma0 = np.cov(projected_data[i0, :].transpose())
    sigma1 = np.cov(projected_data[i1, :].transpose())
    sigma = 0.5 * (sigma0 + sigma1)

    ratio = distance_squared / sigma
    print('----------------RATIO')
    print(ratio)
    return ratio


# EXERCISE 2 - SUBTASK 2.1
# In this task the features and labels are loaded from their
# csv files. Then the features are plotted on the plot in two
# colors based on their label - 0 or 1. The features with label
# 0 are colored in blue and the features with label 1 are colored
# in yellow.
features = np.loadtxt('features.csv')
labels = np.loadtxt('labels.csv')
x = features[:, 0]
y = features[:, 1]
xRed = np.array([])
yRed = np.array([])
xGreen = np.array([])
yGreen = np.array([])
for (x, y, label) in zip(x, y, labels):
    if label == 0:
        xRed = np.append(xRed, x)
        yRed = np.append(yRed, y)
    else:
        xGreen = np.append(xGreen, x)
        yGreen = np.append(yGreen, y)
plt.scatter(xRed, yRed, c='blue', label='Label 0')
plt.scatter(xGreen, yGreen, c='yellow', label='Label 1')
plt.xlabel('Features X')
plt.ylabel('Features Y')
plt.legend()
plt.show()

# EXERCISE 3 - SUBTASK 3.1
# In this task the data for the labels and features is zipped together
# and then is randomly shuffled. After the shuffle, the data is unziped
# and the training set contains 75% while the test set contains 25%. There
# are training and test set for both the features and labels.
shuffle_list_together = np.column_stack((labels, features))
print(features)
print('---------- BEFORE SHUFFLE -----------')
print(shuffle_list_together)
print('---------- AFTER SHUFFLE ------------')
np.random.shuffle(shuffle_list_together)
print(shuffle_list_together)

data_labels_unzip = shuffle_list_together[:, 0]
data_features_unzip = shuffle_list_together[:, [1, 2]]

print('------------- UNZIP --------')
print(data_features_unzip)

print('------------ Train set features --------------')
train_data_features = data_features_unzip[:int(len(data_features_unzip) * 0.75)]
print(train_data_features)
print(train_data_features.size)
print('----------- Test set features -------------')
test_data_features = data_features_unzip[-int(len(data_features_unzip) * 0.25):]
print(test_data_features)
print('---------- Train set labels --------')
train_data_labels = data_labels_unzip[:int(len(data_labels_unzip) * 0.75)]
print(train_data_labels)
print('---------- Test set labels ---------')
test_data_labels = data_labels_unzip[-int(len(data_labels_unzip) * 0.25):]
print(test_data_labels)

# EXERCISE 3 - SUBTASK 3.2
# In this task the RBF kernel classifier is instantiated.
cls = Kfda(kernel='rbf', n_components=1, gamma=0.2)

# EXERCISE 3 - SUBTASK 3.3
# In this task the fitting of the data is done. Then the scores
# for the test data and the training data is calculated for the classifier.
train_cls = cls.fit(train_data_features, train_data_labels)
train_score = cls.score(train_data_features, train_data_labels)
print('Scores train: ', train_score)
test_score = cls.score(test_data_features, test_data_labels)
print('Scores test: ', test_score)

# EXERCISE 4 - SUBTASK 4.1
# In this part, the drawing of the learning curve for the gamma parameter equal
# to 0.2 is done.
draw_learning_curve_different_gamma(train_data_features, train_data_labels, 0.2)

# EXERCISE 4 - SUBTASK 4.2
# In this part, the drawing of the learning curves for all the rest of the gamma
# values is done.
draw_learning_curve_different_gamma(train_data_features, train_data_labels, 0.0001)
draw_learning_curve_different_gamma(train_data_features, train_data_labels, 0.001)
draw_learning_curve_different_gamma(train_data_features, train_data_labels, 0.01)
draw_learning_curve_different_gamma(train_data_features, train_data_labels, 0.1)
draw_learning_curve_different_gamma(train_data_features, train_data_labels, 0.15)
draw_learning_curve_different_gamma(train_data_features, train_data_labels, 0.3)

# EXERCISE 4 - SUBTASK 4.3
# In this part, the prediction of the labels
# based on the gamma value is done.
cls = Kfda(kernel='rbf', n_components=1, gamma=0.01)
cls.fit(train_data_features, train_data_labels)
values = np.array([[0, 1], [2, 3], [7, 0]])
print("Prediction: ")
print(cls.predict(values))

# EXERCISE 5 - SUBTASK 5.1
# In this part, the original features data is transformed.
transformation_features = cls.transform(features)
print(transformation_features)

# EXERCISE 5- SUBTASK 5.2
# In this part, the histograms for each gamma value
# are drawn.
plot_histogram(features, labels, 0.2)
plot_histogram(features, labels, 0.0001)
plot_histogram(features, labels, 0.001)
plot_histogram(features, labels, 0.01)
plot_histogram(features, labels, 0.1)
plot_histogram(features, labels, 0.15)
plot_histogram(features, labels, 0.3)

# EXERCISE 5 - SUBTASK 5.3
# In this part the separation ratio is calculated.
print(separation_ration(transformation_features, labels))

# EXERCISE 6- SUBTASK 6.1
# In this part, the decision boundary is calculated. The
# decision boundary is plotted on the histogram by a black
# point.
i0 = np.where(labels == 0)[0]
i1 = np.where(labels != 0)[0]
print('----------')
print(i0)

u0 = np.mean(transformation_features[i0, :], axis=0)
u1 = np.mean(transformation_features[i1, :], axis=0)
decision_boundary = (u0 + u1) / 2
print('----------DECISION')
print(decision_boundary)
label_one_list = []
label_zero_list = []
for i in range(700):
    if labels[i] == 1:
        label_one_list.append(transformation_features[i, 0])
    else:
        label_zero_list.append(transformation_features[i, 0])
plt.hist(label_zero_list, bins=100, color="blue", label="Label 0")
plt.hist(label_one_list, bins=100, color="yellow", label="Label 1")
plt.hist(decision_boundary, bins=100, color="black", label="Decision")
plt.title("Decision Boundary in Black Point")
plt.legend()
plt.show()

# EXERCISE 6 - SUBTASK 6.2

x = np.linspace(2.5, 13.5,700)
y = np.linspace(-15.5, -4.5,700)
xx, yy = np.meshgrid(x, y, sparse=True)