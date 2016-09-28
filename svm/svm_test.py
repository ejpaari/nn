import sys
sys.path.append("../common")
from class_vis import pretty_picture
from prep_terrain_data import make_terrain_data

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = make_terrain_data()

from sklearn.svm import SVC
clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print "score: %f" % acc
