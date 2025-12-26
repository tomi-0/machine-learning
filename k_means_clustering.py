# Unsupervised earning - we dont provide any clusters
# We train the model on the data we have and find k clusters
# then when we introduces new point we can assign it to a clsuter

# centroids are the centres of the clusters
# and assign them randomly
# we look at individual point and find the nearest centroid
# we then recalculate the centroid and placing it in the middle of the clusters points and then reassign each individual point to the closest centroid interatively
# we stop when the centroids and clusters stop changing

# the algorithm itself
from sklearn.cluster import KMeans

# the function we use to normalise the data # scaling down to 0-1
from sklearn.preprocessing import scale

# dataset of handwritten digits
# we are spotting patterns in data and seeing which category it fits in
from sklearn.datasets import load_digits

digits = load_digits()
#print(digits.data)

import matplotlib.pyplot as plt

# visualize the images
#plt.matshow(digits.images[1], cmap="autumn")
#plt.show()

data = scale(digits.data)

# we specifiy the number of clusters (from 0 to 9)
# and system to determine starting point of clusters
# and how many different initial points/starts

model = KMeans(n_clusters=10, init='random', n_init=10)
model.fit(data)

# we cant just use score since theres not a measure of true or false since this is unsupervised learning
# insteed we could use model.predict and scan in new data to see if it will be put in the correct cluster

