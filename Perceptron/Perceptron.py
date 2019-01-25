import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read the dataset
df = pd.read_csv("data/dataset_8_liver-disorders.csv")

# Prediction
def predict(individual, weights):
    activation = weights[0]
    for i in range(len(individual)):
        activation += weights[i + 1] * individual[i]
    if activation >= 0.0:
        return 2
    else:
        return 1

#Perceptron weights training
def perceptron_train(X_train, y_train, lrn_rate):
    #Randomly chosed weights
    weights = [1.0, 1.6, 1.2, 0.7, 2.1, 0.5, 0.9]
    i=0
    #Run through 5 times to update the weight
    while i < 5:
        i += 1
        sum_error = 0.0
        index = 0
        for individual in X_train:
            prediction = predict(individual, weights)
            error = y_train[index] - prediction
            index += 1
            sum_error += error
            weights[0] = weights[0] + lrn_rate * error
            for i in range(len(individual)):
                weights[i + 1] = weights[i + 1] + lrn_rate * error * individual[i]
    print("weights:", weights)
    return weights

# Perceptron Classifier
def perceptron(X_train, X_test, y_train, y_test, lrn_rate):
    predictions = list()
    weights = perceptron_train(X_train, y_train, lrn_rate)
    for individual in X_test:
        prediction = predict(individual, weights)
        predictions.append(prediction)
    return predictions

def confidence_metric(actual, predicted):
    sum_corr = 0
    sum_err = 0
    for i in range(len(actual)):
        if (predicted[i] - actual[i])**2 < 0.0000001:
            sum_corr += 1
        else:
            sum_err += 1
    return sum_corr / len(actual)

# Compute the mean_accuracy of the algorithm
def mean_accuracy(dataset, compute_time, lrn_rate):
    sum_confidence = 0.0
    for i in range(compute_time):
        # split the data
        dataset.columns = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
        #split dataset
        # x is the columns
        X = dataset.iloc[:, 0:6]
        # y is the last column which is the result
        y = dataset.iloc[:, 6]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        # Data Scaling
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        # predict the results
        predicted = perceptron(X_train, X_test, y_train, y_test, lrn_rate)
        actual_results = y_test
        print("Should be:", actual_results)
        print("Prediction:", predicted)
        accuracy = confidence_metric(actual_results, predicted)
        sum_confidence += accuracy
    return(sum_confidence / compute_time )

# Run perceptron and get the accuracy
mean = mean_accuracy(df, 50, 0.4)
print("Accuracy:", mean* 100 ,"%")
