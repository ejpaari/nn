import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import feature_format, target_feature_split

data_dict = pickle.load(open("../final/final_project_dataset.pkl", "r"))
data_dict.pop("TOTAL", 0)
features = ["salary", "bonus"]
data = feature_format(data_dict, features)

for p in data_dict:
    salary = data_dict[p]["salary"]
    bonus = data_dict[p]["bonus"]
    if salary > 1000000 and salary != "NaN" and bonus > 5000000 and bonus != "NaN":
        print p

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

