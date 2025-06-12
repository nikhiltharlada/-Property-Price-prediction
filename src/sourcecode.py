import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import warnings
warnings.filterwarnings('ignore')
# loading the dataset
data=pd.read_csv(r'D:\work\Property_Price_prediction\src\data_file.csv')
data.head()
data.info()
data.shape
data['total_bedrooms'].unique()
data.isna().sum()
data=data.dropna()
encoder=LabelEncoder()
data['ocean_proximity']=encoder.fit_transform(data['ocean_proximity'])
sns.pairplot(data)
plt.show()
correlation_matrix=data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# simple linear regression
print('simple linear regression')
x=data[['median_income']]
y=data['median_house_value']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('mean squared error of simple linear regression:',mean_squared_error(y_test,y_pred))
print('recall score of simple linear regression:',r2_score(y_test,y_pred))
print('multiple linear regression')
x_multi=data[['households','total_rooms','total_bedrooms','population','median_income','ocean_proximity','longitude', 'latitude','housing_median_age']]
y_multi=data['median_house_value']
x_train_multi,x_test_multi,y_train_multi,y_test_multi=train_test_split(x_multi,y_multi,test_size=0.2,random_state=51)
multi_model=LinearRegression()
multi_model.fit(x_train_multi,y_train_multi)
multi_y_pred=multi_model.predict(x_test_multi)
print('mean squared error of simple linear regression:',mean_squared_error(y_test_multi,multi_y_pred))
print('recall score of simple linear regression:',r2_score(y_test_multi,multi_y_pred))
coefficient=pd.DataFrame(multi_model.coef_,x_multi.columns,columns=['coefficient'])
coefficient.sort_values(by='coefficient',ascending=False)
print(coefficient)