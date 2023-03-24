import pandas as pd
import joblib

# import test_samples
test_samples = pd.read_csv("Labb/test_samples.csv")

# import model
model = joblib.load('Labb/rf_model.pkl')

# choose X and y to predict on test_samples
X = test_samples.drop(columns=['cardio'])
y = test_samples['cardio']

# predict on test_samples
y_pred = model.predict(X)

# print accuracy score
print("Accuracy score: ", model.score(X, y))

# export the predictions to a csv file called "predictions.csv" and it should have the following columns: "Probability class 0", "Probability class 1", "Prediction"

# create a dataframe with the predictions
predictions = pd.DataFrame(model.predict_proba(X), columns=['Probability class 0', 'Probability class 1'])
predictions['Prediction'] = y_pred

# export the predictions to a csv file
predictions.to_csv('Labb/predictions.csv', index=False)


