import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv")
print(data.head())
#Note not to use features with non numerical data.
#So here we have to convert non numeric to numeric data.- preprocessing helps to do that

le = preprocessing.LabelEncoder()
Gender = le.fit_transform(list(data["Gender"]))
Customer_Type = le.fit_transform(list(data["Customer_Type"]))
Age = le.fit_transform(list(data["Age"]))
Subscription_Plan = le.fit_transform(list(data["Subscription_Plan"]))
Price = le.fit_transform(list(data["Price"]))
Ratings = le.fit_transform(list(data["Ratings"]))
Time = le.fit_transform(list(data["Time"]))
Buys = le.fit_transform(list(data["Buys"]))
satis = le.fit_transform(list(data["satisfaction"]))

predict = "satisfaction"

x = list(zip(Gender, Customer_Type, Age, Subscription_Plan, Price, Ratings, Time, Buys, satis))
#Zip creates tuple objects with all diff vals corresponding with lists we give it
y = list(satis)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#print(x_train, y_test) -Data we can pass into classfier to test on

model = KNeighborsClassifier(n_neighbors=7)
#parameter passed is amount of neighbors- hyperparameter or tweak as you continue to train model
#Usually odd so that there will always be a winning class

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["satisfied","neutral or dissatisfied"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ",names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)

print(acc)