import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import io
# Add this import to enable IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt

@st.cache_data
def load_data(uploaded_file):
    """Load data from an uploaded file."""
    if uploaded_file.name.endswith("csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith("tsv"):
        return pd.read_csv(uploaded_file, delimiter='\t')
    else:
        st.error("Unsupported file format.")
        return pd.DataFrame()

@st.cache_data
def load_seaborn_dataset(name):
    """Load dataset from Seaborn."""
    return sns.load_dataset(name)

@st.cache_resource
def get_model(model_selection, hyperparameters):
    """
    Return the appropriate model instance based on user selection and hyperparameters.
    """
    if model_selection == "Linear Regression":
        return LinearRegression()
    elif model_selection == "Decision Tree Regressor":
        return DecisionTreeRegressor()
    elif model_selection == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=hyperparameters.get('n_estimators', 100))
    elif model_selection == "Support Vector Regressor":
        return SVR()
    elif model_selection == "Decision Tree Classifier":
        return DecisionTreeClassifier()
    elif model_selection == "Random Forest Classifier":
        return RandomForestClassifier(n_estimators=hyperparameters.get('n_estimators', 100))
    elif model_selection == "Support Vector Classifier":
        return SVC()
    else:
        return None

def main():
    st.title("Machine Learning Model Builder")

    st.write("""
    Welcome to the Machine Learning Model Builder application!
    This tool helps you to build and evaluate machine learning models on your dataset or on pre-made datasets from Seaborn.
    You can upload your own dataset or choose from Seaborn's datasets, preprocess the data, select and train a model, and evaluate its performance.
    """)

    # Step 2: Upload or choose dataset
    data_source = st.sidebar.selectbox(
        "Choose Data Source",
        ("Upload Dataset", "Use Seaborn Dataset")
    )

    if data_source == "Upload Dataset":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])
        if uploaded_file:
            data = load_data(uploaded_file)
        else:
            st.warning("Please upload a file to proceed.")
            return
    else:
        dataset_name = st.sidebar.selectbox("Choose a Seaborn Dataset", sns.get_dataset_names())
        data = load_seaborn_dataset(dataset_name)

    # Step 5: Display basic data information
    st.write(f"**Dataset:** {dataset_name if data_source == 'Use Seaborn Dataset' else uploaded_file.name}")
    st.write("**Data Head:**")
    st.write(data.head())
    st.write("**Data Shape:**")
    st.write(data.shape)
    st.write("**Data Description:**")
    st.write(data.describe())
    st.write("**Data Info:**")
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write("**Column Names And Datatypes:**")
    st.write(data.dtypes)

    # Step 6: Select features and target
    features = st.multiselect("Select Feature Columns", data.columns)
    target = st.selectbox("Select Target Column", data.columns)

    if target in features:
        st.error("Target column should not be selected as a feature")
        return

    if len(features) == 0:
        st.error("Please select at least one feature column.")
        return

    # Step 7: Problem Type
    is_regression = pd.api.types.is_numeric_dtype(data[target])
    problem_type = st.sidebar.radio(
        "Confirm Problem Type",
        ("Regression", "Classification"),
        index=0 if is_regression else 1
    )

    regression = problem_type == "Regression"

    if regression:
        st.write("**This is a regression problem.**")
        model_types = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"]
    else:
        st.write("**This is a classification problem.**")
        model_types = ["Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"]

    # Sidebar model selection
    model_selection = st.sidebar.selectbox("Select Model", model_types)

    # Add a slider for hyperparameter tuning if a Random Forest model is selected
    hyperparameters = {}
    if model_selection in ["Random Forest Regressor", "Random Forest Classifier"]:
        hyperparameters['n_estimators'] = st.sidebar.slider(
            "Number of Estimators",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )

    # Step 8: Preprocess data
    X = data[features]
    y = data[target]

    # Show data before encoding
    st.write("**Data Before Encoding:**")
    st.write(X.head())

    # Handle missing values
    X_cat = X.select_dtypes(include=['object'])
    X_num = X.select_dtypes(exclude=['object'])

    # Encode categorical features
    encoders = {}
    X_encoded = X_num.copy()
    for col in X_cat.columns:
        encoder = LabelEncoder()
        X_encoded[col] = encoder.fit_transform(X_cat[col].astype(str))
        encoders[col] = encoder

    # Show data after encoding
    st.write("**Data After Encoding:**")
    st.write(X_encoded.head())

    # Combine encoded and numeric features
    X_encoded = pd.concat([X_num, X_encoded], axis=1)

    # Handle missing values in encoded data
    imputer = IterativeImputer()
    X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    # Step 9: Train-Test Split
    test_size = st.slider("Select Test Size", 0.1, 0.9, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Step 10: Initialize Selected Model
    selected_model = get_model(model_selection, hyperparameters)
    
    if selected_model is None:
        st.error("Selected model is not available.")
        return

    # Train and Evaluate the Model
    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)

    metrics = {}
    if regression:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        best_model_name = model_selection
        best_model_score = r2
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "Confusion Matrix": conf_matrix}
        best_model_name = model_selection
        best_model_score = accuracy

    # Step 11: Display Evaluation Metrics
    st.write("**Evaluation Metrics:**")
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):  # Handle confusion matrix
            st.write(f"**{key}:**")
            st.write(value)
        else:
            st.write(f"**{key}:** {value}")
    st.write("")

    # Step 14: Highlight Best Model
    st.write(f"**Best Model:** {best_model_name} with score: {best_model_score}")

    # Step 15: Download Model
    if st.button("Download Best Model"):
        with open('model.pkl', 'wb') as f:
            pickle.dump(selected_model, f)
        st.success("Model saved as model.pkl")

    # Step 16: Make Predictions
    if 'input_data' not in st.session_state:
        st.session_state.input_data = {}
        
    with st.form("input_form"):
        st.write("**Make Predictions:**")
        for feature in features:
            st.session_state.input_data[feature] = st.number_input(f"Input value for {feature}", value=0.0)
        
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        input_data = pd.DataFrame([st.session_state.input_data])
        input_data = input_data.reindex(columns=X_scaled.columns, fill_value=0)
        input_data_scaled = scaler.transform(input_data)
        prediction = selected_model.predict(input_data_scaled)
        st.write("**Prediction Result:**", prediction[0])

if __name__ == "__main__":
    main()