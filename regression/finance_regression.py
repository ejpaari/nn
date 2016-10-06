import sys
sys.path.append("../tools/")
import pickle
from sklearn.linear_model import LinearRegression
from feature_format import feature_format, target_feature_split

dictionary = pickle.load(open("../final/final_project_dataset_modified.pkl", "r") )

features_list = ["bonus", "long_term_incentive"]
data = feature_format(dictionary, features_list, remove_any_zeroes=True)
target, features = target_feature_split(data)

from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "blue"
test_color = "red"

reg = LinearRegression()
reg.fit(feature_train, target_train)

import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color) 
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color) 

plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pass

print "Coef:",reg.coef_
print "Intercept:",reg.intercept_
print "Score:",reg.score(feature_test, target_test)

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()

reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="red") 

plt.savefig("finance_regression.png")
plt.show()
