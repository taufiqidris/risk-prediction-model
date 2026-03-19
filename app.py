import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. PAGE SETUP & UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="ICU Risk Predictor", page_icon="🏥", layout="wide")
st.title("🏥 ICU Mortality Risk Stratification Tool")
st.markdown("""
**Public Health & Clinical Decision Support Interface** This tool utilizes a trained XGBoost model to predict the probability of in-hospital mortality based on initial ICU vitals and lab results. 
*Adjust the key clinical markers in the sidebar to see how patient deterioration affects the risk score in real-time.*
""")

# ==========================================
# 2. CACHED MODEL TRAINING (The "Portfolio Trick")
# ==========================================
@st.cache_resource
def load_and_train_model():
    # Load the data you engineered in the EDA phase
    df = pd.read_csv('notebooks/icu_data_engineered.csv')
    X = df.drop(columns=['in_hospital_death', 'recordid'])
    y = df['in_hospital_death']
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Build the exact pipeline from Notebook 02
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Transform data and sanitize feature names
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=feature_names)
    X_processed.columns = X_processed.columns.str.replace(r"\[|\]|<", "_", regex=True)

    # Train XGBoost
    imbalance_ratio = y.value_counts()[0] / y.value_counts()[1]
    model = xgb.XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        random_state=42, n_estimators=200, learning_rate=0.05, max_depth=4
    )
    model.fit(X_processed, y)
    
    # Store median values of the original dataset to use as defaults for unadjusted sliders
    medians = X.median(numeric_only=True)
    modes = X.mode().iloc[0]
    
    return model, preprocessor, feature_names, medians, modes, X.columns

# Load everything quietly in the background
with st.spinner("Initializing Clinical Pipeline & XGBoost Engine..."):
    xgb_model, preprocessor, feature_names, medians, modes, original_columns = load_and_train_model()

# ==========================================
# 3. SIDEBAR: CLINICAL INPUTS
# ==========================================
st.sidebar.header("Patient Presentation")
st.sidebar.markdown("Adjust key physiological markers:")

# Create interactive sliders for the most important features identified by SHAP in your notebook
age = st.sidebar.slider("Age", 18, 100, int(medians.get('age', 65)))
saps_i = st.sidebar.slider("SAPS-I Score (Severity)", 0, 50, int(medians.get('saps_i', 15)))
lactate = st.sidebar.slider("Lactate (first)", 0.0, 15.0, float(medians.get('lactate_first', 2.0)))
bun = st.sidebar.slider("BUN (first)", 0.0, 100.0, float(medians.get('bun_first', 20.0)))
map_first = st.sidebar.slider("Mean Arterial Pressure (first)", 30, 150, int(medians.get('map_first', 75)))
urine_output = st.sidebar.slider("Urine Output (first)", 0.0, 5000.0, float(medians.get('urineoutput_first', 1500.0)))

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
# Create a dictionary for the new hypothetical patient, starting with the dataset medians/modes
patient_data = {}
for col in original_columns:
    if col in medians.index:
        patient_data[col] = medians[col]
    else:
        patient_data[col] = modes[col]

# Override the defaults with the user's interactive inputs
patient_data['age'] = age
patient_data['saps_i'] = saps_i
patient_data['lactate_first'] = lactate
patient_data['bun_first'] = bun
patient_data['map_first'] = map_first
patient_data['urineoutput_first'] = urine_output

# Convert to dataframe and pass through the pipeline
patient_df = pd.DataFrame([patient_data])
patient_processed = preprocessor.transform(patient_df)
patient_processed = pd.DataFrame(patient_processed, columns=feature_names)
patient_processed.columns = patient_processed.columns.str.replace(r"\[|\]|<", "_", regex=True)

# Generate Prediction
risk_probability = xgb_model.predict_proba(patient_processed)[0][1]

# ==========================================
# 5. DASHBOARD DISPLAY
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Mortality Risk Score")
    # Color code the risk gauge
    if risk_probability < 0.20:
        st.success(f"{risk_probability * 100:.1f}%")
        st.markdown("**Status: Low Risk**")
    elif risk_probability < 0.50:
        st.warning(f"{risk_probability * 100:.1f}%")
        st.markdown("**Status: Elevated Risk** - Monitor hemodynamics closely.")
    else:
        st.error(f"{risk_probability * 100:.1f}%")
        st.markdown("**Status: High Risk** - Indicators of organ dysfunction present.")

with col2:
    st.subheader("Model Explainability (SHAP)")
    st.markdown("How this specific patient's vitals drove the algorithm's decision:")
    
    # Extract SHAP values natively for this single patient
    dtest = xgb.DMatrix(patient_processed.astype(float))
    contribs = xgb_model.get_booster().predict(dtest, pred_contribs=True)
    shap_values_single = contribs[0, :-1] # Drop the bias term
    
    # Plot a clean horizontal bar chart for the top 5 driving factors
    top_indices = np.argsort(np.abs(shap_values_single))[-5:]
    top_features = patient_processed.columns[top_indices]
    top_shaps = shap_values_single[top_indices]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['red' if val > 0 else 'blue' for val in top_shaps]
    ax.barh(top_features, top_shaps, color=colors)
    ax.set_xlabel("Impact on Risk Score (Red = Increases Risk, Blue = Decreases Risk)")
    ax.set_title("Top 5 Clinical Drivers for this Patient")
    st.pyplot(fig)