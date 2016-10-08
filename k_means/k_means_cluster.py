import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import feature_format, target_feature_split

def draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    colors = ["b", "r", "k", "m", "g"]
    for i, p in enumerate(pred):
        plt.scatter(features[i][0], features[i][1], color = colors[pred[i]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for i, p in enumerate(pred):
            if poi[i]:
                plt.scatter(features[i][0], features[i][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

data_dict = pickle.load(open("../final/final_project_dataset.pkl", "r"))
data_dict.pop("TOTAL", 0) # remove an outlier

feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = feature_format(data_dict, features_list)
poi, finance_features = target_feature_split(data)

for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
finance_features = scaler.fit_transform(finance_features)
print "salary 200000.0 and bonus 1000000.0 scaled:",scaler.transform([[200000.0, 1000000.0]])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
pred = kmeans.fit_predict(finance_features)

try:
    draw(pred, finance_features, poi, mark_poi=False, name="clusters.png", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "Prediction object named pred was not found."
