import csv
import math
import random


# Function to load the data from the csv file
def load_csv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    headers = dataset.pop(0)

    # Convert string value to float
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]

    return dataset, headers


# Function to split the dataset into train and test data
def split_data(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = random.randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


# Function to separate the data into classes
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = list()
        separated[vector[-1]].append(vector)
    return separated


# Function to calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Function to calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# Function to calculate the mean, standard deviation and covariance of a list of numbers
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


# Function to calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# Function to calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = row[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities


# Function to predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Function to train the Naive Bayes classifier
def train_naive_bayes(train, test):
    summaries = {}
    for class_value, instances in train.items():
        summaries[class_value] = summarize(instances)
    predictions = list()
    for row in test:
        output = predict(summaries, row)
        predictions.append(output)
    return predictions


# Function to calculate the accuracy of the classifier
def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


# Function to run the Naive Bayes classifier
def run_naive_bayes():
    dataset, headers = load_csv("data/diabetes.csv")
    split = 0.67
    train, test = split_data(dataset, split)
    predictions = train_naive_bayes(train, test)
    accuracy = get_accuracy(test, predictions)
    print("Accuracy: ", accuracy)


run_naive_bayes()
