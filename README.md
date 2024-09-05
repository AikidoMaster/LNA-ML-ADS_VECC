# LNA Design and ML-Based Predictive Maintenance

This project combines the design and simulation of a Low Noise Amplifier (LNA) using Keysight ADS with machine learning-based predictive maintenance.

## Project Overview

The project consists of two main components:

1. LNA Design and Simulation
2. Machine Learning for Predictive Maintenance

## LNA Design and Simulation

The Low Noise Amplifier is designed and simulated using Keysight Advanced Design System (ADS). The LNA is crucial for enhancing weak signals while minimizing noise in various applications such as wireless communication systems.

Key features of the LNA design:
- Optimized for low noise figure
- High gain at the desired frequency
- Good input and output matching

## Machine Learning for Predictive Maintenance with sample simulation data.

Dataset uploaded in the archive(6) folder.

The predictive maintenance component utilizes machine learning techniques to predict equipment failures and optimize maintenance schedules.

### Dataset

The project uses a dataset containing various sensor readings and equipment parameters.

```python
# Load the dataset
df = pd.read_csv('predictive_maintenance.csv')
```

### Data Preprocessing

Data preprocessing steps include:
- Dropping unnecessary columns
- Converting target variable to binary
- Encoding categorical variables
- Feature scaling

```python
# Convert 'Target' to binary
df['Target'] = df['Target'].map({'No Failure': 0, 'Failure': 1})

# Perform one-hot encoding
df = pd.get_dummies(df, columns=['Type'])
```

### Model Training

Multiple machine learning models are trained and evaluated:
1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)

```python
# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```
### Model Evaluation

Models are evaluated using confusion matrices and classification reports.

```python
# Evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))
```

## Model Performance

Here are the performance metrics for each model on the test set:

### Random Forest Classifier

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1799
           1       0.99      0.98      0.98       201

    accuracy                           1.00      2000
   macro avg       0.99      0.99      0.99      2000
weighted avg       1.00      1.00      1.00      2000
```

### Support Vector Machine (SVM)

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1799
           1       0.98      0.97      0.97       201

    accuracy                           0.99      2000
   macro avg       0.99      0.98      0.99      2000
weighted avg       0.99      0.99      0.99      2000
```

### K-Nearest Neighbors (KNN)

```
              precision    recall  f1-score   support

           0       1.00      0.99      1.00      1799
           1       0.94      0.97      0.95       201

    accuracy                           0.99      2000
   macro avg       0.97      0.98      0.97      2000
weighted avg       0.99      0.99      0.99      2000
```

### Model Comparison

The accuracy scores for each model on the test set are:

- Random Forest: 99.75%
- SVM: 99.45%
- KNN: 99.10%

All three models perform exceptionally well, with the Random Forest classifier achieving the highest accuracy.

## Getting Started

1. Clone the repository
2. Install required dependencies
3. Run the Jupyter notebook for machine learning analysis
4. Open the ADS project file for LNA simulation

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Keysight ADS (for LNA simulation)
