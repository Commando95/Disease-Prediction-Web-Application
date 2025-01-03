import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib



heart_data = pd.read_csv("heart.csv")

print(heart_data['target'].value_counts())

x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=101)


model = LogisticRegression()

model.fit(x_train, y_train)


x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy:", training_data_accuracy)

