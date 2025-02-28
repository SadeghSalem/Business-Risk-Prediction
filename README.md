# Business-Risk-Prediction
Certainly! Below is a detailed explanation of all the sections in the Python code:

---

### **1. Importing Necessary Libraries**
```python
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
```
- **pandas**: Used for data manipulation and analysis. Here, it's used to load the dataset into a DataFrame.
- **numpy**: Used for numerical operations, such as handling arrays and matrix operations.
- **train_test_split**: A function from **scikit-learn** that splits data into training and testing sets.
- **LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC, MLPClassifier**: These are machine learning 

models from **scikit-learn** used for classification tasks. Logistic Regression, Decision Tree, Random Forest, Support 

Vector Machine (SVM), and Neural Networks are the models used in this code to predict loan defaults.
- **accuracy_score, precision_score, recall_score, f1_score, confusion_matrix**: These are performance metrics from 

**scikit-learn** to evaluate model performance.
- **matplotlib.pyplot**: Used for plotting charts and graphs to visualize results.
- **seaborn**: Built on top of matplotlib, used for enhanced data visualizations (like heatmaps).
- **SimpleImputer**: Used to handle missing data by filling missing values with a strategy (e.g., mean).
- **LabelEncoder**: Used to convert categorical variables into numeric values, as machine learning models require numeric 

data.

---

### **2. Loading the Dataset**
```python
file_path = '/mnt/data/credit_risk_data.csv'  # Make sure to upload the file in Colab
data = pd.read_csv(file_path)
```
- This section loads the credit risk dataset from a specified file path using **pandas**’ `read_csv` function. The dataset 

will be stored in the `data` DataFrame.

---

### **3. Handling Missing Values**
```python
imputer = SimpleImputer(strategy='mean')
data['person_emp_length'] = imputer.fit_transform(data[['person_emp_length']])
```
- **SimpleImputer** is used to fill in missing values in the `person_emp_length` column. The strategy `'mean'` means that 

missing values will be replaced with the average of the column.
- `fit_transform` is used to apply the imputation to the data and return the transformed version of the column.

---

### **4. Encoding Categorical Features**
```python
label_encoder = LabelEncoder()
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
```
- **LabelEncoder** is used to convert categorical variables (such as `person_home_ownership`, `loan_intent`, `loan_grade`, 

and `cb_person_default_on_file`) into numerical values so that the machine learning algorithms can process them.
- `fit_transform` assigns each unique category a number and transforms the column accordingly.

---

### **5. Splitting Data into Features and Target Variable**
```python
X = data.drop('loan_status', axis=1)
y = data['loan_status']
```
- The dataset is split into **features** (`X`) and the **target variable** (`y`).
- `X` contains all columns except the target variable `loan_status` (whether the loan is default or paid).
- `y` contains only the `loan_status` column, which indicates the actual loan status (defaulted or fully paid).

---

### **6. Splitting Data into Training and Testing Sets**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- The **train_test_split** function splits the data into **training** and **testing** sets.
- `test_size=0.2` indicates that 20% of the data will be used for testing, and 80% will be used for training.
- The `random_state=42` ensures reproducibility by fixing the seed for random operations.

---

### **7. Initializing Models**
```python
logreg_model = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
```
- Here, five different machine learning models are initialized:
  - **Logistic Regression**: A simple model used for binary classification (default vs paid).
  - **Decision Tree**: A non-linear model that splits the data into segments.
  - **Random Forest**: An ensemble model made up of multiple decision trees.
  - **SVM (Support Vector Machine)**: A powerful classifier that finds an optimal hyperplane to separate data points.
  - **Neural Network**: A deep learning model with a specified hidden layer size.
- `random_state=42` is used to ensure reproducibility.

---

### **8. Evaluating the Models**
```python
metrics_dict = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'Confusion Matrix': [],
    'Scatter Plot': []
}
```
- A dictionary `metrics_dict` is initialized to store the performance metrics for each model: Accuracy, Precision, Recall, 

F1-Score, Confusion Matrix, and Scatter Plot.

---

### **9. Training the Models and Calculating Metrics**
```python
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
```
- For each model, the following steps are performed:
  1. The model is trained using the training data (`X_train`, `y_train`).
  2. Predictions are made on the test set (`X_test`).
  3. **Performance metrics** are calculated:
     - **Accuracy**: Measures the percentage of correct predictions.
     - **Precision**: Measures the percentage of correct predictions among all predictions labeled as "default."
     - **Recall**: Measures the percentage of correct predictions among all actual defaults.
     - **F1-Score**: The harmonic mean of Precision and Recall.
     - **Confusion Matrix**: A matrix showing the actual vs. predicted values to help identify misclassifications.
  4. **Scatter Plots** are generated to visualize the model’s predictions versus actual values.

---

### **10. Storing Metrics and Visualizing Results**
```python
metrics_df = pd.DataFrame(metrics_dict, index=model_names)
```
- The **metrics_dict** is converted into a pandas DataFrame (`metrics_df`) for easier visualization and analysis.

```python
import ace_tools as tools; tools.display_dataframe_to_user(name="Model Comparison Metrics", dataframe=metrics_df)
```
- The results are displayed to the user using the `ace_tools` library to show the comparison of metrics across the models.

---

### **11. Comparing Models Using Bar Charts**
```python
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
```
- This section creates a **bar chart** that compares the models across the four evaluation metrics: Accuracy, Precision, 

Recall, and F1-Score.
- `bar_width` determines the width of the bars, and the chart labels the models on the x-axis.

---

### **12. Saving the Preprocessed Dataset**
```python
preprocessed_file_path = '/mnt/data/preprocessed_credit_risk_data.csv'
data_imputed = data.fillna(data.mean())
data_imputed.to_csv(preprocessed_file_path, index=False)
```
- After handling missing values, the preprocessed dataset is saved to a CSV file for further use or downloading.

---

### Summary
This script provides a comprehensive workflow for training and evaluating multiple machine learning models on a credit 

risk dataset. It loads the data, preprocesses it, splits it into training and testing sets, trains various models, 

evaluates their performance using multiple metrics, visualizes the results, and saves the processed data.

Let me know if you need further clarifications or additional explanations!
