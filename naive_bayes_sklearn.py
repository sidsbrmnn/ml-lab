import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Read data from CSV file
data = pd.DataFrame(pd.read_csv("data/news.csv"))

# Map data CATEGORY to numeric values
data["CATEGORY"] = data["CATEGORY"].map({"b": 1, "t": 2, "e": 3, "m": 4})

# Replace nan values with empty string
data = data.fillna("")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data["TITLE"].values, data["CATEGORY"].values, test_size=0.3, random_state=0
)

# Extract features from the training data using CountVectorizer
count_vectorizer = CountVectorizer()
count_train = count_vectorizer.fit_transform(X_train)

# Extract features from the test data using CountVectorizer
count_test = count_vectorizer.transform(X_test)

# Train multinomial Naive Bayes model using CountVectorizer
nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)

# Predict the test set using the multinomial Naive Bayes model
y_pred = nb_classifier.predict(count_test)

# Print the accuracy
print(accuracy_score(y_test, y_pred))

# Print the classification report
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))
