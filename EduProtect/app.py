from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app)


# Load the dataset
data = pd.read_csv('SIH-Dataset.csv')

# Check initial data distribution
print("Initial Dropout Status Distribution:")
print(data['Dropout_Status'].value_counts(normalize=True))

# Visualize the relationship between Socioeconomic Status and Dropout_Status
plt.figure(figsize=(8, 5))
sns.countplot(x='Socioeconomic_Status', hue='Dropout_Status', data=data)
plt.title('Dropout Status by Socioeconomic Status')
plt.xlabel('Socioeconomic Status')
plt.ylabel('Count')
plt.legend(title='Dropout Status', labels=['Did Not Drop Out (0)', 'Dropped Out (1)'])
plt.show()

# Drop unnecessary columns
data = data.drop(columns=["Dropout_Reason"], axis=1)

# Encode the target variable 'Dropout_Status' to binary values (0 or 1)
le = LabelEncoder()
data['Dropout_Status'] = le.fit_transform(data['Dropout_Status'])

# One-hot encode categorical features
X = pd.get_dummies(data.drop('Dropout_Status', axis=1), drop_first=True)
y = data['Dropout_Status']

# Check for class imbalance in the target variable
print("Class Distribution After Encoding:")
print(y.value_counts(normalize=True))

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
print("Class Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts(normalize=True))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

model_file = 'random_forest_model.pkl'
columns_file = 'model_columns.pkl'

# Save the trained model and column names
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')

print("Model trained and saved successfully.")

# Check if model exists, if not, train and save it
if not os.path.exists(model_file) or not os.path.exists(columns_file):
    print("Training model as it doesn't exist yet...")
    train_model()
else:
    # Load the pre-trained model and feature columns
    model = joblib.load(model_file)
    X_columns = joblib.load(columns_file)
    print("Model loaded successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    student_data = request.json
    input_data = pd.DataFrame([student_data])

    # Apply one-hot encoding to the input data
    input_data = pd.get_dummies(input_data)

    # Ensure input columns match the model columns, fill missing columns with 0
    input_data = input_data.reindex(columns=X_columns, fill_value=0)

    # Predict dropout probability
    dropout_probability = model.predict_proba(input_data)[0][1] * 100

    return jsonify({'dropout_probability': f'{dropout_probability:.2f}'})

if __name__ == '__main__':
    app.run(debug=True)
