import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Linear Regression Modal
from sklearn.preprocessing import StandardScaler #used for scaling data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def Format_Data(DataType):
        
        '''
    To make the histogram more like a Gaussian (normal) distribution 
    then we will log all the values
        '''
        '''
        when we will see the histogram we will see  that the histogram is NOT a Gaussian (normal) distribution
        Gaussian (normal) distribution:
        a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean
        '''
        DataType['total_rooms']=np.log(DataType['total_rooms']+1) # to prevent 0 values we are adding 1
        DataType['total_bedrooms']=np.log(DataType['total_bedrooms']+1)  # to prevent 0 values we are adding 1
        DataType['population']=np.log(DataType['population']+1)  # to prevent 0 values we are adding 1
        DataType['households']=np.log(DataType['households']+1)  # to prevent 0 values we are adding 1
        DataType=DataType.join(pd.get_dummies(DataType.ocean_proximity)).drop(['ocean_proximity'],axis=1)
        '''
    Sometimes we will want to add extra features in our training set that makes sense
    to include them in the case of house predication adding features like bedroom radio and  household ratio would add USEFUL features in our model.
    There is a differnce between adding random features and adding useful features and normal features since we know that these  features will play
    an impotannt role  when building our model. 
        '''
        DataType['bedroom_radio']=DataType['total_bedrooms']/DataType['total_rooms']
        DataType['household_rooms']=DataType['total_rooms']/DataType['households']
        return DataType



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

# print(train_data) to print data 

# train_data.hist(figsize=(15,8)) # making a historgram to anylais data 
# # plt.show() #To show all figure
 
# train_data=train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)

# reevualting the correlation
# plt.figure(figsize=(15,8))
# sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')
# plt.show() #To show all figure
train_data=Format_Data(train_data)

# train_data['bedroom_radio']=train_data['total_bedrooms']/train_data['total_rooms']
# train_data['household_rooms']=train_data['total_rooms']/train_data['households']

X_train = train_data.drop('median_house_value', axis=1)
Y_train = train_data['median_house_value']


# The above line actually updated our training data when we updated all the data in our training data
'''
Now we are going to train our data on differnt model and to evalute the results .
'''
reg=LinearRegression()
scaler=StandardScaler()
x_train_s=scaler.fit_transform(X_train) # used for scaling train data
reg.fit(X_train,Y_train)
'''
Now we will repeat the entire process with test_data
'''
test_data=X_test.join(Y_test) # joiing y_test and y_test
test_data=Format_Data(test_data)
# test_data.hist(figsize=(15,8)) # making a historgram to anylais data 
# plt.show() #To show all figure
 

# print(train_data)
# print(test_data)
X_test = train_data.drop('median_house_value', axis=1)
Y_test = train_data['median_house_value']
A=reg.score(X_test,Y_test)
X_test_s=scaler.fit_transform(X_test) # used for scaling test data
print(A,'Linear Regresson Modal')

forest = RandomForestRegressor()
forest.fit(X_train, Y_train)
B = forest.score(X_test, Y_test)
print(B, 'Random Forest Regressor')
'''
Ok alot of new information here
let's dissue what Hyperparameters are:
Hyperparameters help you adjust the parmeters of the modal
such as learing rate and etc 
since Manually selecting the best hyperparameters can be challenging and time-consuming
we use Grid Search Cross-Validation .
Grid Search Cross-Validation help you exhaustive search over a specified hyperparameter grid.
'''

'''
Defining the Hyperparameters below
'''
parms_grid={
    "n_estimators":[3,10,30],
    "max_features":[2,4,6,8],
}

grid_search=GridSearchCV(forest,parms_grid,scoring="neg_mean_squared_error",return_train_score=True)

grid_search.fit(x_train_s,Y_train)
best_forest=grid_search.best_estimator_

print(best_forest.score(X_test_s,Y_test),'best forest')





