import sys
sys.path.append("../common")
from class_vis import pretty_picture, output_image
from prep_terrain_data import make_terrain_data

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn import tree

features_train, labels_train, features_test, labels_test = make_terrain_data()

clf = tree.DecisionTreeClassifier(min_samples_split=50)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "score: %f" % acc

pretty_picture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
