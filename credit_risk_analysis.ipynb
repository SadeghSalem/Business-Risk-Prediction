# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = '/mnt/data/credit_risk_data.csv'  # Make sure to upload the file in Colab
data = pd.read_csv(file_path)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data['person_emp_length'] = imputer.fit_transform(data[['person_emp_length']])

# Convert categorical features to numeric using Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Split the data into features (X) and target variable (y)
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
logreg_model = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# List of models
models = [logreg_model, dt_model, rf_model, svm_model, nn_model]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Neural Network']
colors = ['orange', 'red', 'blue', 'green', 'purple']

# Create a dictionary to store the metrics
metrics_dict = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'Confusion Matrix': [],
    'Scatter Plot': []
}

# Evaluate each model
for model, name in zip(models, model_names):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and calculate metrics
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    metrics_dict['Accuracy'].append(accuracy)
    
    # Precision
    precision = precision_score(y_test, y_pred)
    metrics_dict['Precision'].append(precision)
    
    # Recall
    recall = recall_score(y_test, y_pred)
    metrics_dict['Recall'].append(recall)
    
    # F1-Score
    f1 = f1_score(y_test, y_pred)
    metrics_dict['F1-Score'].append(f1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics_dict['Confusion Matrix'].append(cm)
    
    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, color=colors[model_names.index(name)], alpha=0.5)
    plt.title(f'Scatter Plot - {name}')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics_dict, index=model_names)

# Display the metrics
import ace_tools as tools; tools.display_dataframe_to_user(name="Model Comparison Metrics", dataframe=metrics_df)

# Create a bar chart for comparing accuracy, precision, recall, and F1-Score
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

fig, ax = plt.subplots(figsize=(12, 6))
index = np.arange(len(metrics))
bar_width = 0.15

for i, metric in enumerate(metrics):
    ax.bar(index + i * bar_width, metrics_df[metric], bar_width, label=metric, color=colors)

ax.set_xlabel('Model')
ax.set_ylabel('Values')
ax.set_title('Comparison of Models across Multiple Metrics')
ax.set_xticks(index + 2 * bar_width)
ax.set_xticklabels(model_names)
ax.legend()

plt.tight_layout()
plt.show()

# Save the preprocessed dataset to a CSV file
preprocessed_file_path = '/mnt/data/preprocessed_credit_risk_data.csv'
data_imputed = data.fillna(data.mean())
data_imputed.to_csv(preprocessed_file_path, index=False)

preprocessed_file_path  # Provide the path for downloading
