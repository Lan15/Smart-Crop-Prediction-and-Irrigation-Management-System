
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


#Reading the Excel file
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

#print("Training Data")
#print(train)
#print("Testing Data")
#print(test)

#Dividing the data into training and test set
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.5)

#print("X_Train")
#print(X_train)
#print("X_test")
#print(X_test)
#print(y_train)
#print("Y_test")
#print(y_test)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


#Importing Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()

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
print("The accuracy of this model is: ", a*100)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

 
atemp=input("Enter the temperature")
ah=input("Enter Air Humidity")
pH=input("Enter the pH of the soil")
rain=input("Enter the amount of rainfall")
l=[]
l.append(atemp)
l.append(ah)
l.append(pH)
l.append(rain)
predictcrop=[l]

# Putting the names of crop in a single list
crops=['wheat','mungbean','Tea','millet','maize','lentil','jute','coffee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']
cr='rice'

#Predicting the crop
predictions = clf.predict(predictcrop)

#print(predictions)
count=0
for i in range(0,30):
    if(predictions[0][i]==1):
        c=crops[i]
        count=count+1
        break;
    i=i+1
if(count==0):
    print('The predicted crop is %s'%cr)
else:
    print('The predicted crop is %s'%c)

    
    
    
    
