import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
data=pd.read_csv("/home/ezaan-amin/Documents/Portfolio/House Predication/housing.csv") # reading data from csv
data.dropna(inplace=True) # dropping all null values
# print(data.info()) # seeing if there are no null values
# spliting the data into two  format  using train_test_split
# train_test_split is a method where we split the data on two parts one is used to train and the other is to test the data
X=data.drop(['median_house_value'], axis=1) # training  this is the training  dataset
Y=data['median_house_value'] #  this is the test data we will use to test our model
'''
so what the below code is doing is actually we are giving 20% of our data
to the test dataset . Later we will evalute our model on test dataset
'''
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
train_data=X_train.join(Y_train) # joiing x_train and y_train
'''
So this next part is a liitle confsuing for me I will  try my best
to get it right.
import seaborn as sns
is a libary build upoon matplotlib it is basically used for  anylasis data
in graphic data 
'''
# train_data.hist(figsize=(15,8)) # making a historgram to anylais data 
# plt.show() #To show all figure
'''
next thing is we are gonna find the  correlation  between data.
what is  correlation ?
connection or relationship between two or more facts, numbers, etc. 
what this avtually mean is higher the number mean higher the realtionship b/w
two fields mean the higher the relationship
for example if we see the figure
you will see a type of 2d array that the best way to describe it
but notice the the columns where two fields are comman you will
see that the number is 1 simce almost all the numbers are same

for this project we need to see which of the value have a higher relationship with 
median_icome  since that is the  field we are gonna to predict 
'''
# plt.figure(figsize=(15,8))
# sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')
# plt.show() #To show all figure
'''
when we will see the histogram we will see  that the histogram is NOT a Gaussian (normal) distribution
Gaussian (normal) distribution:
a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean
'''
'''
To make the histogram more like a Gaussian (normal) distribution 
then we will log all the values
'''
# print(train_data) to print data 
train_data['total_rooms']=np.log(train_data['total_rooms']+1) # to prevent 0 values we are adding 1
train_data['total_bedrooms']=np.log(train_data['total_bedrooms']+1)  # to prevent 0 values we are adding 1
train_data['population']=np.log(train_data['population']+1)  # to prevent 0 values we are adding 1
train_data['households']=np.log(train_data['households']+1)  # to prevent 0 values we are adding 1
train_data.hist(figsize=(15,8)) # making a historgram to anylais data 
# plt.show() #To show all figure
'''
if you look at the training data then there is a column called ocean_proximity the problem with ocean_proximity is the fact 
a string value and we can't compute string into our model.
so we will first have to convert into int but we can't just assign a rammdon number
no
so we will have to one hot encode them.
One hot encoding is a technique that we use to represent categorical variables as numerical values in a machine learning model.
'''
# pd.get_dummies(train_data.ocean_proximity) # One hot encoding  
ocean_proximity_data=pd.get_dummies(train_data.ocean_proximity)   

train_data=train_data.join(ocean_proximity_data) # joining dataset
# reevualting the correlation
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')
plt.show() #To show all figure





