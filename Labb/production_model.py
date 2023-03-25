import pandas as pd
import joblib

# import test_samples
test_samples = pd.read_csv("Labb/test_samples.csv")

# import model
model = joblib.load('Labb/rf_model.pkl')

# use trained model to predict on test_samples
y = test_samples['cardio']

y_pred = model.predict_proba(y)

# print accuracy score
print("Accuracy score: ", model.score(test_samples, y_pred))

# create a dataframe with the predictions
predictions = pd.DataFrame(model.predict_proba(test_samples), columns=['Probability class 0', 'Probability class 1'])
predictions['Prediction'] = y_pred

# export the predictions to a csv file
predictions.to_csv('Labb/predictions.csv', index=False)


