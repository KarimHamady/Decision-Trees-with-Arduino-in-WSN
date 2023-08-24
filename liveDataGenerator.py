import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import parameters
import numpy as np
import serial
import matplotlib.pyplot as plt

test_ratio = 0.2

# ------------------------------------------------------------------------
# Training phase
data = pd.read_excel("data.xlsx", sheet_name="humidity")
# Create DataFrame
df = pd.DataFrame(data, columns=["Humidity", "Output"])
print(data, df)
# Split the data into training and testing sets
X = df[["Humidity"]]
y = df["Output"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

# Fit a decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# ------------------------------------------------------------------------

# Testing phase

# Define the serial port and baud rate
port = 'COM13'
baud_rate = 9600

# Open the serial connection
ser = serial.Serial(port, baud_rate)
alarmRaised = False
sensorData1 = []
sensorData2 = []

while True:
    while len(sensorData1) < test_ratio*240:
        value1, value2 = ser.readline().decode('utf-8').split(' ')
        postValue1 = int(value1.rstrip())
        postValue2 = int(value2.rstrip()[:-1])
        sensorData1.append(postValue1)
        sensorData2.append(postValue2)
    y_pred1 = model.predict(sensorData1)
    y_pred2 = model.predict(sensorData2) 

sensorData1 = np.array(sensorData1)
sensorData2 = np.array(sensorData2)
sensorData1 = sensorData1.reshape(-1, 1)
sensorData2 = sensorData2.reshape(-1, 1)
print(sensorData1, sensorData2)



# f = open("sensorData.txt", "a")
# for sD in range(len(sensorData1)):
#     f.write(f'{sensorData1[sD][0]} {y_pred1[sD]} {sensorData2[sD][0]} {y_pred2[sD]} \n')
# f.close()

# Calculate accuracy of the model
# accuracy = accuracy_score(y_sensorData1, y_pred)
# print("Accuracy:", accuracy)
