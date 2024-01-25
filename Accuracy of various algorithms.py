import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv('cpdata.csv')
print(data.head(1))


#Creating dummy variable for target i.e label
label= pd.get_dummies(data.label).iloc[: , 1:]
data= pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
#print('The data present in one row of the dataset is')
#print(data.head(1))
train=data.iloc[:, 0:4].values
test=data.iloc[: ,4:].values




X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.2)


#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy using KNN",accuracy*100)



#Importing Random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators=100, random_state=42)

#Fitting the classifier into training set
clf.fit(X_train,y_train)
pred=clf.predict(X_test)

#clf=DecisionTreeRegressor()

#Fitting the classifier into training set
clf.fit(X_train,y_train)
pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a=accuracy_score(y_test,pred)
print("Accuracy using Random forest classifier ", a*100)



#Importing Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
clf1=DecisionTreeRegressor()

#Fitting the classifier into training set
clf1.fit(X_train,y_train)
pred=clf1.predict(X_test)


from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a1=accuracy_score(y_test,pred)
print("Accuracy using decision tree regressor ", a1*100)

#Importing Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf2=DecisionTreeClassifier()

#Fitting the classifier into training set
clf2.fit(X_train,y_train)
pred=clf2.predict(X_test)

#clf=DecisionTreeRegressor()


from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a2=accuracy_score(y_test,pred)
print("Accuracy using decision tree classifier", a2*100)
