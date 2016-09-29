import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree

features_train, features_test, labels_train, labels_test = preprocess()

print "number of features: %d" % len(features_train[0])

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "score: %f" % acc
