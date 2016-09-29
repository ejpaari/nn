import sys
sys.path.append("../tools/")
sys.path.append("../common")
from time import time
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC
#clf = SVC(kernel="linear")
clf = SVC(C=10000.0, kernel='rbf')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
