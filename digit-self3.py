from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf

nc = 10  # Number of classes

# Load the training data
df_train = pd.read_csv("train3.csv")
X_train = df_train.iloc[:, 1:785].values
y_train = df_train.iloc[:, 0:1].values

# Load the test data
df_test = pd.read_csv('test1.csv')
X_test = df_test.iloc[:, 0:784].values

# Standardize the data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# One-hot encode the target variable for training
y_train = tf.one_hot(y_train[:, 0], depth=nc)

# Build the model
model = Sequential()
model.add(Dense(20, input_dim=784, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(nc, activation="softmax"))  # Use softmax for multiclass classification
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=10)

# Predict on the test data
# ypred = model.predict(X_test)
ypred = np.argmax(model.predict(X_test), axis=1)

dataset = pd.DataFrame(ypred) 
#DATA FRAME CONVERTS IT INTO KIND OF AN EXCEL SHEET FOR US TO PERFORM NUMPY OPERATIONS
#check the output
#print(df)
dataset.to_csv('Prediction')
