import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
import pandas as pd
nc = 10 # Number of classes

#0: airplane
#1: automobile
#2: bird
#3: cat
#4: deer
#5: dog
#6: frog
#7: horse
#8: ship
#9: truck

(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()

Xtrain = Xtrain.astype('float32')
Xtrain = Xtrain[0:20000,:] / 255.0
Xtest = Xtest.astype('float32')
Xtest = Xtest / 255.0
ytrainEnc = tf.one_hot(ytrain[0:20000:,0], depth=nc) 
ytestEnc = tf.one_hot(ytest[:,0], depth=nc)

model = Sequential()
model.add(Conv2D(32,(3,3),activation="relu",kernel_initializer='he_uniform',padding='same',input_shape=(32,32,3)))
model.add(Conv2D(32,(3,3),activation="relu",kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation="relu",kernel_initializer='he_uniform',padding='same'))
model.add(Conv2D(64,(3,3),activation="relu",kernel_initializer='he_uniform',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation="relu",kernel_initializer='he_uniform',padding='same'))
model.add(Conv2D(128,(3,3),activation="relu",kernel_initializer='he_uniform',padding='same'))

model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation="relu",kernel_initializer='he_uniform'))
model.add(Dense(10,activation="softmax"))
opt = SGD(learning_rate = 0.001, momentum = 0.9)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(Xtrain,ytrainEnc,epochs=100)

ypred = model.predict(Xtest)
ypred = np.argmax(ypred,axis=1)
# score = accuracy_score(ypred,ytest)
# print('Accuracy score is',100*score,'%')
df = pd.DataFrame(ypred) 
#DATA FRAME CONVERTS IT INTO KIND OF AN EXCEL SHEET FOR US TO PERFORM NUMPY OPERATIONS
#check the output
#print(df)
df.to_csv('Prediction')