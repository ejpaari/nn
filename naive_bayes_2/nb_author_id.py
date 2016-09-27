import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB

features_train, features_test, labels_train, labels_test = preprocess()

gnb = GaussianNB()
gnb.fit(features_train, labels_train)
gnb.predict(features_test)
print gnb.score(features_test, labels_test)
