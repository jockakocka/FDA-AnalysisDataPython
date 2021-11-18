import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#import data
dataset1 = np.loadtxt('dataset1_noCluster15.csv', skiprows=1, delimiter=',')
dataset2 = np.loadtxt('dataset2_noCluster2.csv', skiprows=1, delimiter=',')
dataset3 = np.loadtxt('dataset3_noCluster2.csv', skiprows=1, delimiter=',')
dataset4 = np.loadtxt('dataset4_noCluster2.csv', skiprows=1, delimiter=',')

#extract type labels for each dataset
types_1 = dataset1[:,2]
types_2 = dataset2[:,2]
types_3 = dataset3[:,2]
types_4 = dataset4[:,2]

#delete type column from each dataset
np.delete(dataset1, 2, axis=1)
np.delete(dataset2, 2, axis=1)
np.delete(dataset3, 2, axis=1)
np.delete(dataset4, 2, axis=1)

#data processing
processing_scalar = StandardScaler()
processing_1 = processing_scalar.fit_transform(dataset1)
processing_2 = processing_scalar.fit_transform(dataset2)
processing_3 = processing_scalar.fit_transform(dataset3)
processing_4 = processing_scalar.fit_transform(dataset4)

#coloring based on type
colors_1 = []
colors_2 = []
colors_3 = []
colors_4 = []
for type in types_1:
    if type == 1:
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
        colors_1.append("black")
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

for type in types_2:
    if type == 1:
        colors_2.append("yellow")
    else:
        colors_2.append("blue")

for type in types_3:
    if type == 1:
        colors_3.append("yellow")
    else:
        colors_3.append("blue")

for type in types_4:
    if type == 1:
        colors_4.append("yellow")
    else:
        colors_4.append("blue")

#plot dataset1
x_1 = processing_1[:,0]
y_1 = processing_1[:,1]
plt.scatter(x_1, y_1,c=colors_1)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Data Set 1 - No. Cluster 15')
type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
type2_patch = mpatches.Patch(color='blue', label='Class of type 2')
type3_patch = mpatches.Patch(color='red', label='Class of type 3')
type4_patch = mpatches.Patch(color='green', label='Class of type 4')
type5_patch = mpatches.Patch(color='grey', label='Class of type 5')
type6_patch = mpatches.Patch(color='black', label='Class of type 6')
type7_patch = mpatches.Patch(color='brown', label='Class of type 7')
type8_patch = mpatches.Patch(color='orange', label='Class of type 8')
type9_patch = mpatches.Patch(color='purple', label='Class of type 9')
type10_patch = mpatches.Patch(color='pink', label='Class of type 10')
type11_patch = mpatches.Patch(color='olive', label='Class of type 11')
type12_patch = mpatches.Patch(color='navy', label='Class of type 12')
type13_patch = mpatches.Patch(color='cyan', label='Class of type 13')
type14_patch = mpatches.Patch(color='gold', label='Class of type 14')
type15_patch = mpatches.Patch(color='magenta', label='Class of type 15')
plt.legend(
            handles=[type1_patch, type2_patch, type3_patch, type4_patch, type5_patch, type6_patch, type7_patch,
                     type8_patch,
                     type9_patch, type10_patch, type11_patch, type12_patch, type13_patch, type14_patch,
                     type15_patch])
plt.show()

#plot dataset2
x_2 = processing_2[:,0]
y_2 = processing_2[:,1]
plt.scatter(x_2, y_2,c=colors_2)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Data Set 2 - No. Cluster 2')
type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
type2_patch = mpatches.Patch(color='blue', label='Class of type 2')
plt.legend(handles=[type1_patch, type2_patch])
plt.show()

#plot dataset3
x_3 = processing_3[:,0]
y_3 = processing_3[:,1]
plt.scatter(x_3, y_3,c=colors_3)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Data Set 3 - No. Cluster 2')
type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
type2_patch = mpatches.Patch(color='blue', label='Class of type 2')
plt.legend(handles=[type1_patch, type2_patch])
plt.show()

#plot dataset4
x_4 = processing_4[:,0]
y_4 = processing_4[:,1]
plt.scatter(x_4, y_4,c=colors_4)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Data Set 4 - No. Cluster 2')
type1_patch = mpatches.Patch(color='yellow', label='Class of type 1')
type2_patch = mpatches.Patch(color='blue', label='Class of type 2')
plt.legend(handles=[type1_patch, type2_patch])
plt.show()
