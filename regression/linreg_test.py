import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ages_net_worths import age_net_worth_data

ages_train, ages_test, net_worths_train, net_worths_test = age_net_worth_data()

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

net_worth = reg.predict([27])
print "Net worth for 27yo:",net_worth
print "Coef:",reg.coef_
print "Intercept:",reg.intercept_

print "Test score:",reg.score(ages_test, net_worths_test)
print "Training score:",reg.score(ages_train, net_worths_train)

plt.scatter(ages_train, net_worths_train, color="blue")
plt.scatter(ages_test, net_worths_test, color="green")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.xlabel("age")
plt.ylabel("net worth")
plt.savefig("linreg_test.png")
plt.show()

