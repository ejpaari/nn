import sys
sys.path.append("../common")

from class_vis import pretty_picture, output_image
from prep_terrain_data import make_terrain_data

import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

features_train, labels_train, features_test, labels_test = make_terrain_data()

"""
The training data (features_train, labels_train) have both "fast" and "slow"
points mixed together--separate them so we can give them different colors
in the scatterplot and identify them visually.
"""
grade_fast = [features_train[i][0] for i in range(0, len(features_train)) if labels_train[i]==0]
bumpy_fast = [features_train[i][1] for i in range(0, len(features_train)) if labels_train[i]==0]
grade_slow = [features_train[i][0] for i in range(0, len(features_train)) if labels_train[i]==1]
bumpy_slow = [features_train[i][1] for i in range(0, len(features_train)) if labels_train[i]==1]

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
#clf = AdaBoostClassifier(SVC(kernel="poly", C=10000.0), algorithm="SAMME", n_estimators=200)
#clf = AdaBoostClassifier(GaussianNB(), algorithm="SAMME", n_estimators=200)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "score: %f" % acc

try:
    pretty_picture(clf, features_test, labels_test)
except NameError:
    pass
