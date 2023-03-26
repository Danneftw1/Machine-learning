import pandas as pd
import joblib

# import test_samples
test_samples = pd.read_csv("Labb/test_samples.csv")

# import model
model = joblib.load('Labb/rf_model.pkl')

# use trained model to predict on test_samples
X_test = test_samples.drop('cardio', axis=1)
y_test = test_samples['cardio']

y_pred = model.predict_proba(X_test)

# print accuracy score
print("Accuracy score: ", model.score(X_test, y_test))

# create a dataframe with the predictions
predictions = pd.DataFrame(y_pred, columns=['Probability class 0', 'Probability class 1'])
predictions['Prediction'] = model.predict(X_test)

# export the predictions to a csv file
predictions.to_csv('Labb/predictions.csv', index=False)


