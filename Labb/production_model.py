import pandas as pd
import joblib

# import test_samples
test_samples = pd.read_csv("Labb/test_samples.csv")

# import model
model = joblib.load('Labb/svm_model.pkl')

# choose X and y to predict on test_samples
X = test_samples.drop(columns=['cardio'])
y = test_samples['cardio']

# predict on test_samples
y_pred = model.predict(X)

# print accuracy score
print("Accuracy score: ", model.score(X, y))
