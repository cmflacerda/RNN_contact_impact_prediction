# Setup
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

def dataMatrix(data,step):
    X,Y = [], []
    for i in range(len(data)-step):
        d = i + step
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

# Preparing the data
torque_inte = np.array([0.056,0.194,0.389,0.583,0.778,1,1.222,
                        1.389,1.556,1.75,1.917,2.139,2.333,2.583,
                        2.889,3.167,3.444,3.694,4,4.306,4.5,4.722,
                        4.944,5.083,5.306,5.583,5.694,5.806,5.861,
                        5.944,6.028,6.111,6.25,6.389])

torque_col = np.array([0.056,0.222,0.417,0.583,0.806,1,1.167,
                       1.389,1.667,2,2.333,2.611,2.944,3.194,
                       3.444,3.75,4.028,4.333,4.75,5.111,5.444,
                       5.667,5.917,6.167,6.361,6.556,6.722,6.917,
                       7.083,7.306,7.528,7.75,8,8.278])

q_dot_inte = np.array([43.311,43.133,43.267,43.267,43.311,43.044,42.978,
                       42.756,42.556,42.311,42.089,42.000,41.756,41.644,41.822,
                       41.644,41.422,41.178,40.978,40.822,40.956,40.822,
                       40.578,40.444,40.333,40.022,39.933,39.778,39.6,
                       39.489,39.333,39.222,39.156,39.089])
q_dot_inte = q_dot_inte[::-1]

q_dot_col = np.array([28.847,28.958,29.139,29.042,29.194,28.944,28.819,
                      29.069,29.139,29.069,28.833,28.931,29.028,29.194,
                      29.417,29.597,29.667,29.819,29.944,29.931,30.014,
                      30.222,30.028,29.944,29.875,29.986,30.25,30.083,
                      30.181,29.972,29.972,30.111,30.25,30.153])

N = len(torque_inte)
Tp = 30
df = pd.DataFrame(torque_inte)
df_one = pd.DataFrame(torque_col)
df.head()

values = df.values
values_one = df_one.values
train, test = values[0:Tp,:], values[Tp:N,:]
train_one, test_one = values_one[0:Tp,:], values_one[Tp:N,:]

step = 2
train = np.append(train, np.repeat(train[-1],step))
test = np.append(test,np.repeat(test[-1],step))

train_one = np.append(train_one, np.repeat(train[-1],step))
test_one = np.append(test_one,np.repeat(test[-1],step))

train_cp_x, train_cp_y = dataMatrix(train,step)
test_cp_x, test_cp_y = dataMatrix(test,step)

train_cp_one_x, train_cp_one_y = dataMatrix(train_one,step)
test_cp_one_x, test_cp_one_y = dataMatrix(test_one,step)

train_cp_x_f = np.reshape(train_cp_x, (train_cp_x.shape[0], 1, train_cp_x.shape[1]))
test_cp_x_f = np.reshape(test_cp_x, (test_cp_x.shape[0], 1, test_cp_x.shape[1]))

train_cp_one_x_f = np.reshape(train_cp_one_x, (train_cp_one_x.shape[0], 1, train_cp_one_x.shape[1]))
test_cp_one_x_f = np.reshape(test_cp_one_x, (test_cp_one_x.shape[0], 1, test_cp_one_x.shape[1]))

# Creating the model to be used in contact identification
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(1,step), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()

# Creating the model to be used in impact identification
model_one = Sequential()
model_one.add(SimpleRNN(units=40, input_shape=(1,step), activation='relu'))
model_one.add(Dense(8, activation='relu'))
model_one.add(Dense(1))
model_one.compile(loss='mean_squared_error', optimizer='rmsprop')
model_one.summary()

# Training the contact identification model
model.fit(train_cp_x_f, train_cp_y, batch_size=2, epochs=100, verbose=2)
train_predicted = model.predict(train_cp_x_f)
test_predicted = model.predict(test_cp_x_f)
predicted = np.concatenate((train_predicted,test_predicted), axis=0)
trainScore = model.evaluate(train_cp_x_f, train_cp_y, verbose=0)

# Training the impact identification model
model_one.fit(train_cp_one_x_f, train_cp_one_y, batch_size=2, epochs=100, verbose=2)
train_predicted_one = model_one.predict(train_cp_one_x_f)
test_predicted_one = model_one.predict(test_cp_one_x_f)
predicted_one = np.concatenate((train_predicted_one,test_predicted_one), axis=0)
trainScore_one = model.evaluate(train_cp_one_x_f, train_cp_one_y, verbose=0)

# Plotting general result
# The blue curve is the contact torque from dataset
# The orange curve is the contact torque predicted by the RNN
# The green curve is the impact torque from dataset
# The red curve is the impact torque predicted by the RNN
# The vertical red line separates training and test data
index = df.index
index_one = df_one.index
plt.plot(index, values)
plt.plot(index, predicted)
plt.axvline(index[Tp], c="r")

plt.plot(index_one, values_one)
plt.plot(index_one, predicted_one)
plt.axvline(index[Tp], c="r")
plt.show()

# Plotting the prediction of interaction made by the collision model
test_predicted_two = model_one.predict(train_cp_x_f)
plt.plot(index_one[0:30], test_predicted_two)
plt.plot(index_one, predicted_one)
plt.plot(index_one, predicted)
plt.show()

# Plotting the prediction of collision made by the interaction model
test_predicted_three = model.predict(train_cp_one_x_f)
plt.plot(index[0:30], test_predicted_three)
plt.plot(index, predicted_one)
plt.plot(index, predicted)
plt.show()

# First attempt to predict the variation in torque using the variation in velocity
delta_q_dot_inte = []
delta_q_dot_col = []
delta_torque_inte = []
for i in range(len(q_dot_inte) - 1):
    delta_q_dot_inte.append((q_dot_inte[i] + q_dot_inte[i+1]) / 2)
    delta_q_dot_col.append((q_dot_col[i] + q_dot_col[i+1]) / 2)
    delta_torque_inte.append((torque_inte[i] + torque_inte[i+1]) / 2)
delta_q_dot_inte.insert(0, q_dot_inte[0])
delta_q_dot_col.insert(0, q_dot_col[0])
delta_torque_inte.insert(0, torque_inte[0])
delta_q_dot_inte = np.array(delta_q_dot_inte)
delta_q_dot_col = np.array(delta_q_dot_col)
delta_torque_inte = np.array(delta_torque_inte)

delta_q_dot_inte = np.append(delta_q_dot_inte, np.repeat(train[-1],step))
delta_q_dot_inte_cp,_ = dataMatrix(delta_q_dot_inte,step)
delta_q_dot_inte_cp_f = np.reshape(delta_q_dot_inte_cp, (delta_q_dot_inte_cp.shape[0], 1, delta_q_dot_inte_cp.shape[1]))

# Creating the model of delta_torque prediction using a delta_velocity
model_three = Sequential()
model_three.add(SimpleRNN(units=600, input_shape=(1,step), activation='relu'))
model_three.add(Dense(8, activation='relu'))
model_three.add(Dense(1))
model_three.compile(loss='mean_squared_error', optimizer='rmsprop')
model_three.summary()

# Training the model of delta_torque prediction using a delta_velocity
model_three.fit(delta_q_dot_inte_cp_f, delta_torque_inte, batch_size=1, epochs=1000, verbose=2)
train_predicted_three = model_one.predict(delta_q_dot_inte_cp_f)

plt.plot(index, delta_torque_inte)
plt.plot(index, train_predicted_three)
plt.show()