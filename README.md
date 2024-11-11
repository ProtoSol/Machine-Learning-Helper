# Machine Learning Helper

## Live Link

[Machine Learning Helper](https://ml-model-builder.streamlit.app/)

## Overview

This Streamlit application provides a user-friendly interface for building and evaluating machine learning models. It allows users to upload their own datasets or use pre-built Seaborn datasets, perform data preprocessing, train various models, and evaluate their performance.

## Features

- **Data Input Options**:
  - Upload custom datasets (CSV, XLSX, TSV formats)
  - Use built-in Seaborn datasets
- **Data Analysis**:
  - View basic data statistics and information
  - Analyze data types and missing values
  - Visualize data distributions
- **Preprocessing Capabilities**:
  - Automatic handling of missing values using IterativeImputer
  - Label encoding for categorical variables
  - Feature scaling using StandardScaler
- **Model Selection**:
  - Regression Models:
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Support Vector Regressor
  - Classification Models:
    - Decision Tree Classifier
    - Random Forest Classifier
    - Support Vector Classifier
- **Model Evaluation**:
  - Regression Metrics: MSE, RMSE, MAE, R²
  - Classification Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **Additional Features**:
  - Hyperparameter tuning for Random Forest models
  - Model download functionality
  - Real-time predictions on user input

## Dependencies

- streamlit
- seaborn
- pandas
- numpy
- scikit-learn
- matplotlib
- pickle

## Usage

1. Run the Streamlit app:
```bash
streamlit run ml_app.py
```

2. Use the sidebar to:
   - Choose data source (upload file or use Seaborn dataset)
   - Select model type
   - Adjust hyperparameters (if applicable)

3. In the main interface:
   - Select feature columns and target variable
   - Adjust test size for train-test split
   - View model performance metrics
   - Download trained model
   - Make real-time predictions

## Data Requirements

- Dataset should be in CSV, XLSX, or TSV format
- Data should be properly formatted with headers
- Target variable should be clearly identifiable
- Numeric data for regression tasks
- Categorical or numeric data for classification tasks

## Model Building Process

1. Data Loading and Exploration
2. Feature Selection
3. Problem Type Identification (Regression/Classification)
4. Data Preprocessing
   - Missing Value Imputation
   - Categorical Variable Encoding
   - Feature Scaling
5. Model Training and Evaluation
6. Results Visualization and Model Download

## Disclaimer

This application is designed for educational and experimental purposes. Model performance may vary depending on the quality and nature of the input data. Always validate models thoroughly before using them in production environments.