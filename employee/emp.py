import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset to set input and output variables
dataset = pd.read_csv('train.csv');
X = dataset.iloc[:, 0:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

X_train_before_scaling = X_train

# Feature Scaling
'''
Feature saling is important step it will change numbers like 174, 175 to small unit 
like 0.578, 0.687
It standardize data
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train_after_scaling = X_train

#================================
# Part 2 - Now Let's Make ANN! 
#================================

# importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# adding the input layer and the first hidden layer
classifier.add( Dense(units=3, kernel_initializer='uniform', activation='relu', input_dim=2 ) )

# adding second hidden layer
classifier.add( Dense(units=2, kernel_initializer='uniform', activation='relu') )

# adding the output layer
classifier.add( Dense(units=1, kernel_initializer='uniform', activation='sigmoid') )

# Compiling the ANN
classifier.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

# Fitting the ANN to the Training set

classifier.fit( X_train, Y_train, batch_size=10, epochs=100)


# Predicting the test set result
datasetTest = pd.read_csv('test.csv');
testData = datasetTest.iloc[:, 0:2].values

testData = sc.transform(testData)
testData = classifier.predict(testData)
testData = (testData > 0.5)