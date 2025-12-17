import numpy as np
# for visulatisations
import matplotlib.pyplot as plt
# for using model
from sklearn.linear_model import LinearRegression
# fnction to split our data into training and testing data
from sklearn.model_selection import train_test_split

# we make up the data in this example
# creates a numpy array

# we need to reshape them from being horizontal because scikit learn needs it
# in a vertical format so we need to transpose it
time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1, 1)
scores = np.array([56, 83, 47, 93, 47, 82, 45, 78, 55, 67, 57, 4, 12]).reshape(-1, 1)

# next we need to train the model
model = LinearRegression()

# to find the optimised line we do:
model.fit(time_studied, scores)

# Now we want to visulaise our data points
plt.scatter(time_studied, scores)

# shows best fit line for some data on already trained data
# first arg gives us 100 values from 0 to 70 for x value
# our y value we use model predictions
plt.plot(np.linspace(0,70,50).reshape(-1, 1), model.predict(np.linspace(0,70,50).reshape(-1, 1)), 'red')

# we can get a prediction for a single value 
print(model.predict(np.array([56]).reshape(-1, 1)))

# set y limit
plt.ylim(0, 100)
# display plot
plt.show()

# we can also then test our model
times = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1, 1)
scored = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]).reshape(-1, 1)

# we use 20% of our data for testing model
time_train, time_test, score_train, score_test = train_test_split(times, scored, test_size=0.3)

model2 = LinearRegression()
# only train on our training set
model2.fit(time_train, score_train)

# we have data we havent fed to the model
# we want to see how well it can predict these values

# we can improve this (i.e. avoid -ves) by increasing test size
print(model2.score(time_test, score_test))

plt.scatter(time_train, score_train)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)))
plt.show()
