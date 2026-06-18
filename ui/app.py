import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction App")
st.markdown("This app predicts the likelihood of heart disease based on clinical features.")

# Load models and preprocessing objects
try:
    model_path = os.path.join(os.path.dirname(__file__), '../models/final_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
    features_path = os.path.join(os.path.dirname(__file__), '../models/selected_features.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)
except FileNotFoundError:
    st.error("Model files not found. Please train the models first by running the pipeline.")
    st.stop()

# Creating the UI layout
st.sidebar.header("Patient Input Features")

# To accept inputs, we need to know the original feature names. 
# We'll use Streamlit widgets for common heart disease dataset features.
# Based on UCI Heart Disease dataset standard columns:
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4], help="1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic")
trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
chol = st.sidebar.slider("Serum Cholestoral in mg/dl (chol)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [1, 0])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", 70, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [1, 0])
oldpeak = st.sidebar.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.2, 1.0)
slope = st.sidebar.selectbox("Slope of the peak exercise ST segment (slope)", [1, 2, 3])
ca = st.sidebar.slider("Number of major vessels (0-3) colored by flourosopy (ca)", 0, 3, 0)
thal = st.sidebar.selectbox("Thal", [3, 6, 7], help="3 = normal; 6 = fixed defect; 7 = reversable defect")

# Create a dataframe for the input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Since the model expects dummy variables if they were one-hot encoded, 
# and selected features via RFE. We need to construct a df that matches X before scaling.
# We will create a dummy df with all original columns the model might expect
# We'll need to re-create the dummy logic.
categorical_cols = ['cp', 'restecg', 'slope', 'thal']

# Let's create a template dataframe with all possible categories to ensure correct shape
template_df = pd.DataFrame(columns=[
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    'cp_2', 'cp_3', 'cp_4', 'restecg_1', 'restecg_2', 'slope_2', 'slope_3', 'thal_6', 'thal_7'
])

# For one-hot encoding
input_dict = {
    'age': age, 'sex': sex, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
    'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'ca': ca
}
for col in ['cp_2', 'cp_3', 'cp_4', 'restecg_1', 'restecg_2', 'slope_2', 'slope_3', 'thal_6', 'thal_7']:
    input_dict[col] = 0

if f'cp_{cp}' in input_dict: input_dict[f'cp_{cp}'] = 1
if f'restecg_{restecg}' in input_dict: input_dict[f'restecg_{restecg}'] = 1
if f'slope_{slope}' in input_dict: input_dict[f'slope_{slope}'] = 1
if f'thal_{thal}' in input_dict: input_dict[f'thal_{thal}'] = 1

# Make sure all columns in scaler's feature_names_in_ are present
# We will just construct a DF matching the order of scaler.feature_names_in_
final_input_df = pd.DataFrame([input_dict], columns=scaler.feature_names_in_)

# Scale the data
scaled_input = pd.DataFrame(scaler.transform(final_input_df), columns=scaler.feature_names_in_)

# Select only the features the model was trained on
model_input = scaled_input[selected_features]

st.subheader("Prediction")

if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(model_input)
    probability = model.predict_proba(model_input)[0][1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
            
    with col2:
        st.metric(label="Disease Probability", value=f"{probability * 100:.1f}%")
        
    # Feature contributions using basic logic (since it's a random forest, SHAP would be better but we keep it simple)
    st.subheader("Data Visualization")
    st.write("This application helps in early detection by utilizing machine learning. Your inputs are passed through an optimized Random Forest model.")
    
st.markdown("---")
st.markdown("Developed as part of the Sprints Machine Learning Pipeline Project.")
