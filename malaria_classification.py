import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib

# Step-1: Load DataSet

dataFrame = pd.read_csv("csv/dataset.csv")
print(dataFrame.head())

# Step-2: Split data into training and test set

x = dataFrame.drop(["Label"], axis=1)
y = dataFrame["Label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step-3: Build a model

model = RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(x_train, y_train)
joblib.dump(model, "rf_malaria_100_5")

# Step-4: Make predictions and get classification report

predictions = model.predict(x_test)

print(metrics.classification_report(predictions, y_test))
print(model.score(x_test, y_test))

