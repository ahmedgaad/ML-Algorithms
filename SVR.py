import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics

# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('car.csv')

# inspecting the first 5 rows of the dataframe
car_dataset.head()

# checking the number of rows and columns
car_dataset.shape

# getting some information about the dataset
# car_dataset.info()

# checking the number of missing values
car_dataset.isnull().sum()

# checking the distribution of categorical data
# print(car_dataset.Fuel_Type.value_counts())
# print(car_dataset.Seller_Type.value_counts())
# print(car_dataset.Transmission.value_counts())


# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


car_dataset.head()

X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']

# print(X)
# print(Y)

#Splitting Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

#Model Training
#################################### loading the Support Vector Regression model
svr_model = SVR()
svr_model.fit(X_train,Y_train)

# prediction on Training data
training_data_prediction = svr_model.predict(X_train)

# evaluation
training_error = svr_model.score(X_train, Y_train)
print('Training Error: ',training_error)


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

# prediction on Testing data
test_data_prediction = svr_model.predict(X_test)

# evaluation
testing_error = svr_model.score(X_test, Y_test)
print('Testing Error: ',testing_error)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()