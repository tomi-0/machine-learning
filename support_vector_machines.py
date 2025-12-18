# --------------- What are Support Vector Machines ------------------

# very poweful tools for classifying data using support vectors
# in soemc ases may outperform neural networks
# e.g. handwritten digits , normal neural networks most of the time perform worse except CNNs

# ----------------------- How are they trained --------------------------

# we train a linear function to split the data in the most optimal way
#we want a line furthest away from all the points

# to find the line we use support vectors
# we draw a parallel line from another horisontal line on which is
# on the edge of our data from the current linear model

# the space between is called a margin area (free space between base line and points)
# the goal is to find the line that splits point in 2 classes w/ 
# the largest margin area


# ----------------------- Kernal Functions --------------------------

# Add an additional dimension to our data
# so we can use a hyperplane to separate the data

# we arent allowed to add new data 
# so we use the existing features to create a new feature (redundant)

# we ue predefined kernal functions that are optimised for doing this operation on the data


# ----------------------- Soft Margin --------------------------

# sort of like a tolerance
# we can defined a soft margin of a value to allow a misclassification
# up to a certain number of data points to get a better model
  
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# svc = support vector classifier
from sklearn.svm import SVC

# just to compare accuracy
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

X = data.data
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# C = soft margin
clf = SVC(kernel='linear', C=3)
clf.fit(x_train, y_train)

print(f"SVC: {clf.score(x_test, y_test)}")


clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)

# score always changes as the 20% test set is always chosen at random
# can use e.g. random_state=23 to keep it fixed
print(f"KNN: {clf2.score(x_test, y_test)}")