import pandas as pd

# import test_samples
test_samples = pd.read_csv('../Labb/test_samples.csv')

# import model
from sklearn.externals import joblib
model = joblib.load('svm_model.pkl')

# make a prediction on the 100 test samples with the model
predictions = model.predict(test_samples)

# print the predictions
print(predictions)

# save the predictions to a csv file
pd.DataFrame(predictions).to_csv('predictions.csv')