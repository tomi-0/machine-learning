# amount of classes/labelled clusters shouldnt be divisible by k

# importing breast cancer data set from scikit learn
# includes parameters about cancer and tumors andd classfies them
# as either malignanat(bad) or benign(good)
# function to load breast cancer data
from sklearn.datasets import load_breast_cancer

# calculates class based on nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

# we will first take a look at the data by running the function
data = load_breast_cancer()
print(data)

# prints features
print(data.feature_names)
# prints class labels/names (malignant or benign)
print(data.target_names)

# avtual data feature values
# print(np.array(data.data))
# avtual classes
print(np.array(data.target))

# split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

# now we define our classifier and we specify how many neighbours
clf = KNeighborsClassifier(n_neighbors=3) 

# we train on our train set
clf.fit(x_train, y_train)

# we then want to evaluate our model using test set
print(clf.score(x_test, y_test))

X = [d[20] for d in x_train]

# not really an accurate representation
# since im plotting classification value against a single feature
plt.scatter(X, y_train)
plt.show()


# could predict a class for a list of features using
# clf.predict([...])