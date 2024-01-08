import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder,StandardScaler

train=pd.read_csv('train (2).csv')
test = pd.read_csv('test (1).csv')

# Assuming 'data' is your DataFrame and 'column_name' is the column with string values
#le = LabelEncoder()
#train['Street'] = le.fit_transform(train['Street'])
#train['MSZoning'] = le.fit_transform(train['MSZoning'])
#train['LotShape'] = le.fit_transform(train['LotShape'])
#train['Alley'] = le.fit_transform(train['Alley'])
#train['Utilities'] = le.fit_transform(train['Utilities'])
#train['LandSlope'] = le.fit_transform(train['LandSlope'])
#train['LotConfig'] = le.fit_transform(train['LotConfig'])
#train['LandContour'] = le.fit_transform(train['LandContour'])
#train['Neighborhood'] = le.fit_transform(train['Neighborhood'])
#train['Id'] = le.fit_transform(train['Id'])

#from sklearn.preprocessing import LabelEncoder

# Assuming 'data' is your DataFrame
le = LabelEncoder()

# Apply label encoding to all columns
train = train.apply(lambda col: le.fit_transform(col) if col.dtype == 'O' else col)

# Note: 'O' is the dtype for Python objects, which includes strings

train = train.ffill() 

print(train.shape)
Xtrain = train.iloc[:,0:80].values
Ytrain = train.iloc[:,80:81].values

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
#Ytrain = scaler.fit_transform(Ytrain)

reg = LinearRegression()
reg.fit(Xtrain,Ytrain)

ytrain_predict = reg.predict(Xtrain)
r2 = r2_score(Ytrain,ytrain_predict)
print("r2 = ",r2)
#rmse = np.sqrt(mean_squared_error(Ytrain,ytrain_predict))
#print("RMSE = ",rmse)

Xtest = train.iloc[:,0:80].values
Xtest = scaler.fit_transform(Xtest)
Ytest = reg.predict(Xtest)

df = pd.DataFrame(Ytest) 
#DATA FRAME CONVERTS IT INTO KIND OF AN EXCEL SHEET FOR US TO PERFORM NUMPY OPERATIONS
#check the output
#print(df)
df.to_csv('Prediction', index=False)