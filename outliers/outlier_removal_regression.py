import random
import numpy
import matplotlib.pyplot as plt
import pickle
from outlier_cleaner import outlier_cleaner

ages = pickle.load(open("practice_outliers_ages.pkl", "r"))
net_worths = pickle.load(open("practice_outliers_net_worths.pkl", "r"))

ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

try:
    plt.plot(ages, reg.predict(ages), color="blue")
    print "Coef:",reg.coef_
    print "Intercept:",reg.intercept_
    print "Test score:",reg.score(ages_test, net_worths_test)
except NameError:
    pass

plt.scatter(ages, net_worths)
plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlier_cleaner(predictions, ages_train, net_worths_train)
except NameError:
    print "Your regression object doesn't exist, or isn't name reg can't make predictions to use in identifying outliers."

if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
        print "Coef:",reg.coef_
        print "Intercept:",reg.intercept_
        print "Test score:",reg.score(ages_test, net_worths_test)
    except NameError:
        print "You don't seem to have regression imported/created, or else your regression object isn't named reg. Either way, only draw the scatter plot of the cleaned data."
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()
else:
    print "outlier_cleaner() is returning an empty list, no refitting to be done"

