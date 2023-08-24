import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Data in the given format
data = open("sensorData.txt", "r")

# Extracting values from column 1 and column 2
column0_values = []
column1_values = []
column2_values = []
column3_values = []

for entry in data:
    values = entry.split(' ')
    column0_values.append(int(values[0]))
    column1_values.append(int(values[1]))
    column2_values.append(int(values[2]))
    column3_values.append(int(values[3]))
y_sensorData1 = [1 if(i > 45 or i < 30) else 0 for i in column0_values]
y_sensorData2 = [1 if(i > 45 or i < 30) else 0 for i in column2_values]

accuracy1 = accuracy_score(y_sensorData1, column1_values)
print("Accuracy1:", accuracy1)
accuracy2 = accuracy_score(y_sensorData2, column3_values)
print("Accuracy2:", accuracy2)
alarms = [90 if column1_values[i] and column3_values[i] else 0 for i in range(len(column0_values)) ]
sensor1Fault = [80 if (column0_values[i] > 45 or column0_values[i] < 30) and (column2_values[i] > 30 and column2_values[i] < 45) else 0 for i in range(len(column0_values)) ]
sensor2Fault = [70 if (column2_values[i] > 45 or column2_values[i] < 30) and (column0_values[i] > 30 and column0_values[i] < 45) else 0 for i in range(len(column0_values)) ]


# Plotting the data
x_values = range(len(column0_values))
plt.plot(x_values, column0_values, label='Sensor 1')
plt.plot(x_values,column2_values, label='Sensor 2')
plt.plot(x_values,sensor1Fault, label='Sensor 1 Fault')
plt.plot( x_values,sensor2Fault, label='Sensor 2 Fault')
plt.plot(x_values,alarms, label='Alarm')

plt.ylabel('RH')
plt.legend()
plt.show()