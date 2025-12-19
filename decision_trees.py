# --------------------- What are Decision Trees ------------------------

# We will usually have some data with several features and an outcome
# we try to build a decision tree model to predict the right outcome
# we give the model this data and other parameters and will end up in a classification

#                       Root
#                      /     \
#                   Sunny    Rain 
#                  /     \  /     \
#   We end up choosing Y/N brnches (in this case) to then classify the data

# --------------------- How to train Decision Trees ------------------------

# We define a root node which asks a randomquestion for some feature
# then we split the possibilities into different branches
# we then split the data into different branches depending on the value of that feature

# we check how many of the data rows end up being a Final Yes or No and keep track of that number
# if 100% of data points are in the same category e.g. Yes we can end decision tree and give a final prediciton
# otherwise we can ask another question for a different feature

# --------------------- Random Forest Classification ------------------------

# A decision tree can be random because of the order of the features
# can lead to hhuge differences in the output

# for Random Forest Classifcation we create multiple tree e.g. 50/100/200 and
# train all of them on the same data
# we then feed a new test example into all of them and choose the collective result of the forest - minimises risk of misclassification


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X = data.data
Y = data.target 

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

clf2 = RandomForestClassifier() # we will use default values of 100 estimators and etc 
clf2.fit(x_train, y_train)

print(f"Decision Tree: {clf.score(x_test, y_test)}")
print(f"Random Forest: {clf2.score(x_test, y_test)}")
