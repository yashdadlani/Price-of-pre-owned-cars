import os
import pandas as pd
import numpy as np
import seaborn as sns
os.chdir("D:\Datasets")
sns.set(rc={'figure.figsize':(11.7,8.27)})
cars_data = pd.read_csv("cars_sampled.csv")
cars = cars_data.copy() # deeep copy
col = ['name','dateCrawled','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)
cars.drop_duplicates(keep='first',inplace=True)
yearwise_count =cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration', y = 'price',scatter=True,fit_reg=False,
            data=cars)
#variable price
price_count=cars['price'].value_counts().sort_index()
sns.displot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)

power_count=cars['powerPS'].value_counts().sort_index()
sns.displot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS', y = 'price',scatter=True,fit_reg=False,
            data=cars)
sum(cars['price']>500)
sum(cars['price']<10)
#Working Range of Data
cars = cars[(cars.yearOfRegistration <= 2018)
            &(cars.yearOfRegistration>=1950)
            &(cars.price >= 100) & (cars.price <=500) &(cars.powerPS>= 10)
            &(cars.powerPS<=500)]
# Variable Reduction
#combine year of registration and month of registration
cars['monthOfRegistration']/=12
cars['Age'] = (2018-cars['yearOfRegistration']) + cars['monthOfRegistration']
cars['Age'] = round(cars['Age'],2)
cars['Age'].describe()
cars = cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#Visualization
sns.distplot(cars['Age'])
sns.boxplot(y = cars['Age'])
sns.distplot(cars['price'])
sns.boxplot(y = cars['price'])
sns.distplot(cars['powerPS'])
sns.boxplot(y = cars['powerPS'])
sns.regplot(x='Age', y = 'price',scatter=True,fit_reg=False,
            data=cars)

#powerPS vs price
sns.regplot(x='powerPS', y = 'price',scatter=True,fit_reg=False,
            data=cars)

cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x = 'seller',data=cars)

cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x = 'abtest',data=cars)
sns.boxplot(x = 'abtest', data=cars)

cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x = 'vehicleType',data=cars)
sns.boxplot(x = 'vehicleType',y = 'price', data=cars)

cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x = 'gearbox',data=cars)
sns.boxplot(x = 'gearbox',y = 'price', data=cars)

cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x = 'kilometer',data=cars)
sns.boxplot(x = 'kilometer',y = 'price', data=cars)
cars['kilometer'].decribe()
sns.distplot(cars['kilometer'],bins = 8,kde = False)
sns.regplot(x='kilometer', y = 'price',scatter=True,fit_reg=False,
            data=cars)


cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x = 'model',data=cars)
sns.boxplot(x = 'model',y = 'price', data=cars)

cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x = 'fuelType',data=cars)
sns.boxplot(x = 'fuelType',y = 'price', data=cars)

cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x = 'brand',data=cars)
sns.boxplot(x = 'brand',y = 'price', data=cars)

cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x = 'notRepairedDamage',data=cars)
sns.boxplot(x = 'notRepairedDamage',y = 'price', data=cars)


#Removing Insignificant Variables

col = ['seller','offerType','abtest']
cars = cars.drop(columns=col,axis=1)
cars_copy = cars.copy()

#Correlation
cars_select1 = cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

#removing missing values
cars_omit = cars.dropna(axis=0)


cars_omit = pd.get_dummies(cars_omit,drop_first=True)
#Model Building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RandomForestRegressor
from sklearn.metrics import mean_squared_error

x1 = cars_omit.drop(['price'],axis='columns',inplace=False)
y1 = cars_omit['price']
prices = pd.DataFrame({"1. Before":y1,"2. After":np.log(y1)})
prices.hist()

y1 = np.log(y1)
# splitting data into test and train

X_train, X_test, y_train, y_test= train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape, X_test.shape, y_train.shape,y_test.shape)

#baseline model to use mean value of test data
base_pred = np.mean(y_test)

#Root Mean Square Error
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
#Linear Regression with removed data
lgr = LinearRegression(fit_intercept=True)
model_lin1 = lgr.fit(X_train,y_train)
cars_predictions_lin1 = lgr.predict(X_test)
#Calculate MSE and RMSE
lin_mse1 = mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)

#R squared value
r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)


#residual analysis
residuals1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residuals1,scatter=True,
            fit_reg=False,data=cars)
residuals1.describe()

# Random Forest Algorithm
#model parameters
rf = RandomForestRegressor(n_estimators = 100,max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

#Model
model_rf1 = rf.fit(X_train,y_train)
cars_predictions_rf1 = rf.predict(X_test)

#Calculating MSE and RMSE

rf_mse1 = mean_squared_error(y_test, cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)

#Model building with imputed data
cars_imputed = cars.apply(lambda x: x.fillna(x.median()) \
                   if x.dtype=='float' else \
                       x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()
cars_imputed = pd.get_dummies(cars_imputed,drop_first=True)

x2 = cars_imputed.drop(['price'],axis='columns',inplace=False)
y2 = cars_imputed['price']
prices = pd.DataFrame({"1. Before":y2,"2. After":np.log(y2)})
prices.hist()

y2 = np.log(y2)
X_train1, X_test1, y_train1, y_test1= train_test_split(x2,y2,test_size=0.3,random_state=3)
                                                       

base_pred = np.mean(y_test1)
base_pred = np.repeat(base_pred,len(y_test1))
base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test1, base_pred))    
         
lgr2 = LinearRegression(fit_intercept=True)
model_lin2 = lgr.fit(X_train1,y_train1)
cars_predictions_lin2 = lgr.predict(X_test1)
#Calculate MSE and RMSE
lin_mse2 = mean_squared_error(y_test1,cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)

#R squared value
r2_lin_test2=model_lin2.score(X_test1,y_test1)
r2_lin_train2=model_lin2.score(X_train1,y_train1)

rf2 = RandomForestRegressor(n_estimators = 100,max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

#Model
model_rf2 = rf.fit(X_train1,y_train1)
cars_predictions_rf2 = rf.predict(X_test1)

#Calculating MSE and RMSE

rf_mse2 = mean_squared_error(y_test1, cars_predictions_rf2)
rf_rmse2 = np.sqrt(rf_mse2)

# Rsquared value
r2_rf_test2 = model_rf2.score(X_test1,y_test1)
r2_rf_train2 = model_rf2.score(X_train1,y_train1)

