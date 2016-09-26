from prep_terrain_data import make_terrain_data
from class_vis import pretty_picture, output_image
from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf

features_train, labels_train, features_test, labels_test = make_terrain_data()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[i][0] for i in range(0, len(features_train)) if labels_train[i]==0]
bumpy_fast = [features_train[i][1] for i in range(0, len(features_train)) if labels_train[i]==0]
grade_slow = [features_train[i][0] for i in range(0, len(features_train)) if labels_train[i]==1]
bumpy_slow = [features_train[i][1] for i in range(0, len(features_train)) if labels_train[i]==1]

clf = classify(features_train, labels_train)

pretty_picture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
