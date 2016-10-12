import sys
import pickle
sys.path.append("../tools/")

from feature_format import feature_format, target_feature_split
from tester import dump_classifier_and_data, test_classifier

def add_ratio_feature(data_dict, key, new_feature, dividend, divisor):
    try:
        data_dict[key][new_feature] = data_dict[key][dividend] / data_dict[name][divisor]
    except TypeError:
        data_dict[key][new_feature] = "NaN"
    except:
        print "Unexpected error:", sys.exc_info()[0]

features_list = ["poi", "salary", "bonus", "total_payments", "total_stock_value"]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers
outliers = ["TOTAL", "THE TRAVEL AGENCY IN THE PARK"]
for outlier in outliers:
    data_dict.pop(outlier)

### Create new features
for name in data_dict:    
    add_ratio_feature(data_dict, name, "from_poi_ratio", "from_poi_to_this_person", "to_messages")
    add_ratio_feature(data_dict, name, "to_poi_ratio", "from_this_person_to_poi", "from_messages")

features_list += ["from_poi_ratio", "to_poi_ratio"]

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = feature_format(my_dataset, features_list, sort_keys = True)
labels, features = target_feature_split(data)

### Classify
### Name your classifier clf for easy export below.
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest

selection = SelectKBest()
rfc = RandomForestClassifier()
pipeline = Pipeline([('features', selection), ('classifier', rfc)])

parameters = {'features__k': [5, 'all'],
              'classifier__n_estimators': [50, 100, 200],
              'classifier__min_samples_split': [2, 4, 6], 
              'classifier__criterion': ['entropy', 'gini'],
              'classifier__class_weight': ['balanced_subsample', 'auto', None],
              'classifier__max_depth': [2, 4, 6]
}

clf = GridSearchCV(pipeline, parameters, scoring='recall')
clf.fit(features, labels)
test_classifier(clf.best_estimator_, my_dataset, features_list)

### Dump the classifier
dump_classifier_and_data(clf, my_dataset, features_list)
