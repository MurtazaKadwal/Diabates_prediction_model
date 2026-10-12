
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="GlucoSense | Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD SAVED FILES
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "diabetes_model.joblib")
    features = joblib.load(BASE_DIR / "diabetes_features.joblib")
    scaler = joblib.load(BASE_DIR / "diabetes_scaler.joblib")

except Exception as e:
    st.error(f"Unable to load the model files: {e}")
    st.info(
        "Make sure all three JOBLIB files are in the same "
        "folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0b1120;
        color: #e5edf7;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #101a2e;
        border-right: 1px solid #263650;
    }

    [data-testid="stSidebar"] * {
        color: #e5edf7;
    }

    .hero {
        background: linear-gradient(
            135deg, #142640 0%, #102b38 100%
        );
        border: 1px solid #29465a;
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #42e8c3;
        text-transform: uppercase;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .hero h1 {
        color: #f4f8ff;
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 14px 0;
    }

    .hero p {
        color: #b8c8dc;
        font-size: 15px;
        line-height: 1.7;
        margin: 0;
        max-width: 760px;
    }

    .section-heading {
        color: #f4f8ff;
        font-size: 21px;
        font-weight: 750;
        margin: 10px 0 5px 0;
    }

    .section-description {
        color: #96a9c2;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .panel {
        background: #111d31;
        border: 1px solid #263852;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 20px;
    }

    .result-card {
        background: linear-gradient(
            135deg, #13273a, #102f34
        );
        border: 1px solid #2a5b5c;
        border-radius: 20px;
        padding: 28px;
        margin-top: 12px;
        margin-bottom: 20px;
    }

    .result-label {
        color: #9eb4c9;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .result-title {
        color: #f5faff;
        font-size: 28px;
        font-weight: 800;
        margin: 12px 0;
    }

    .result-copy {
        color: #bdcddd;
        font-size: 14px;
        line-height: 1.7;
    }

    .info-card {
        background: #111d31;
        border: 1px solid #263852;
        border-radius: 15px;
        padding: 18px;
        min-height: 105px;
    }

    .info-title {
        color: #92a8c2;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .info-value {
        color: #f4f8ff;
        font-size: 21px;
        font-weight: 800;
    }

    div[data-testid="stForm"] {
        background: #111d31;
        border: 1px solid #263852;
        border-radius: 18px;
        padding: 22px;
    }

    div[data-testid="stForm"] label {
        color: #dce7f5 !important;
        font-weight: 600;
    }

    div[data-testid="stForm"] input {
        background-color: #0b1425;
        color: #f5faff;
    }

    div[data-testid="stForm"] [data-baseweb="input"] {
        background-color: #0b1425;
        border-radius: 9px;
    }

    .stButton button,
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(
            90deg, #14b8a6, #2dd4bf
        );
        color: #071522 !important;
        font-weight: 800;
        border: none;
        border-radius: 10px;
        min-height: 48px;
        transition: 0.2s ease;
    }

    .stButton button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #5eead4;
        border: none;
        box-shadow: 0 5px 22px rgba(45, 212, 191, 0.2);
    }

    div[data-testid="stMetric"] {
        background: #111d31;
        border: 1px solid #263852;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9eb2ca;
    }

    .footer {
        color: #758ba5;
        text-align: center;
        font-size: 12px;
        border-top: 1px solid #263650;
        padding-top: 18px;
        margin-top: 35px;
    }

    .stAlert {
        border-radius: 12px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 22px;
        }

        .panel {
            padding: 16px;
        }

        .result-card {
            padding: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-size:34px;'>🩺</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='margin-bottom:0;'>GlucoSense</h2>",
        unsafe_allow_html=True
    )

    st.caption("Diabetes Prediction System")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Prediction Dashboard", "ℹ️ About the Model"],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### Model information")
    st.markdown("**Algorithm:** Decision Tree")
    st.markdown("**Selection:** GridSearchCV")
    st.markdown("**CV folds:** 5")
    st.markdown("**Test accuracy:** 76.62%")

    st.divider()

    st.caption(
        "Educational machine learning project. "
        "Not a medical diagnostic tool."
    )


# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">HEALTH INTELLIGENCE • ML PROJECT</div>
    <h1>Understand your health.<br>Explore your risk.</h1>
    <p>
        Enter the health measurements below to generate a prediction
        from a trained machine learning model. This tool is intended
        for educational exploration and does not diagnose diabetes.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
if page == "ℹ️ About the Model":

    st.markdown(
        '<div class="section-heading">About this project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    This application uses a trained **Decision Tree classifier**
    to demonstrate a binary classification workflow for diabetes data.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="panel">
            <h3>Model details</h3>
            <p>Algorithm: Decision Tree Classifier</p>
            <p>Splitting criterion: Entropy</p>
            <p>Maximum depth: 3</p>
            <p>Minimum samples per leaf: 4</p>
            <p>Hyperparameter tuning: GridSearchCV</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="panel">
            <h3>Saved assets</h3>
            <p><code>diabetes_model.joblib</code></p>
            <p><code>diabetes_features.joblib</code></p>
            <p><code>diabetes_scaler.joblib</code></p>
            <p>Inputs are scaled using the saved transformer before prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    st.warning(
        "The reported test accuracy is from your model evaluation. "
        "It does not represent clinical validation or guarantee "
        "the reliability of an individual prediction."
    )

    st.stop()


# --------------------------------------------------
# DASHBOARD SUMMARY CARDS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">MODEL ALGORITHM</div>
        <div class="info-value">Decision Tree</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">INPUT FEATURES</div>
        <div class="info-value">08 Features</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">TEST ACCURACY</div>
        <div class="info-value">76.62%</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">Patient measurements</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter all eight values. Use measurements from a reliable source '
    'where possible.'
    '</div>',
    unsafe_allow_html=True
)

with st.form("diabetes_prediction_form"):

    st.markdown("#### Personal information")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Number of pregnancies",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
            help="Enter 0 if not applicable."
        )

    with col2:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

    st.markdown("#### Clinical measurements")

    col1, col2 = st.columns(2)

    with col1:
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=500,
            value=110,
            step=1,
            help="Use a glucose measurement in mg/dL."
        )

        blood_pressure = st.number_input(
            "Diastolic blood pressure (mm Hg)",
            min_value=0,
            max_value=250,
            value=70,
            step=1
        )

        skin_thickness = st.number_input(
            "Skin thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            step=1
        )

        insulin = st.number_input(
            "Insulin (mu U/mL)",
            min_value=0,
            max_value=1000,
            value=80,
            step=1
        )

    with col2:
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1,
            format="%.1f"
        )

        pedigree = st.number_input(
            "Diabetes pedigree function",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.01,
            format="%.2f",
            help="Enter the value used by your dataset."
        )

        st.info(
            "Enter the original measurements, not manually "
            "standardized values."
        )

    submitted = st.form_submit_button(
        "✨ Generate Prediction",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION LOGIC AND RESULTS
# Only run after the form is submitted
# --------------------------------------------------
if submitted:

    # Map form values to the original dataset feature names.
    input_values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }

    # Ensure the saved feature names match the app's inputs.
    missing_features = [
        feature for feature in features
        if feature not in input_values
    ]

    if missing_features:
        st.error(
            "Some saved model features do not match the form inputs: "
            + ", ".join(missing_features)
        )
        st.stop()

    # Create input data in exactly the saved feature order.
    input_df = pd.DataFrame(
        [[input_values[feature] for feature in features]],
        columns=features
    )

    try:
        # Apply the saved scaler used during training.
        input_scaled = scaler.transform(input_df)

        # Predict the class.
        prediction = int(model.predict(input_scaled)[0])

        # Obtain probability for Outcome 1, if supported.
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(
                    probabilities[classes.index(1)]
                )

    except Exception as e:
        st.error(f"Unable to generate prediction: {e}")
        st.stop()

    # ----------------------------------------------
    # FRIENDLY RESULT DISPLAY
    # ----------------------------------------------
    st.divider()

    st.markdown(
        '<div class="section-heading">Prediction Results</div>',
        unsafe_allow_html=True
    )

    if prediction == 0:

        st.markdown("""
        <div class="result-card" style="
            border-color: #2a8b72;
            background: linear-gradient(135deg, #132d35, #10352f);
        ">
            <div class="result-label" style="color: #5eead4;">
                PREDICTION RESULT
            </div>
            <div class="result-title">
                Low Risk of Diabetes
            </div>
            <div class="result-copy">
                Based on the information provided, the model classified
                this result as Outcome 0, indicating a lower predicted
                likelihood of diabetes in this dataset.
                This result does not rule out diabetes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card" style="
            border-color: #b45353;
            background: linear-gradient(135deg, #38212b, #351d28);
        ">
            <div class="result-label" style="color: #fca5a5;">
                PREDICTION RESULT
            </div>
            <div class="result-title">
                High Risk of Diabetes
            </div>
            <div class="result-copy">
                Based on the information provided, the model classified
                this result as Outcome 1, indicating a higher predicted
                likelihood of diabetes in this dataset.
                Please consult a qualified healthcare professional
                for appropriate testing and evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------
    # ADDITIONAL RESULT INFORMATION
    # ----------------------------------------------
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        result_label = (
            "Low Risk of Diabetes"
            if prediction == 0
            else "High Risk of Diabetes"
        )

        st.metric(
            "Prediction",
            result_label
        )

    with result_col2:
        if probability is not None:
            st.metric(
                "Model Probability for Outcome 1",
                f"{probability * 100:.1f}%"
            )
        else:
            st.metric(
                "Model Probability",
                "Unavailable"
            )

    if probability is not None:
        st.markdown("#### Model Probability Visualization")

        st.progress(
            max(0.0, min(1.0, probability)),
            text=(
                "Estimated probability assigned to Outcome 1 "
                "by the model — not a clinical risk score."
            )
        )

    with st.expander("Review Submitted Measurements"):
        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Educational prediction only. This model is not a diagnostic "
        "tool, and its output should not replace professional medical advice."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    GLUCOSENSE • DIABETES PREDICTION PROJECT<br><br>
    Built with Python, Scikit-learn and Streamlit<br>
    For educational purposes only — not for medical diagnosis.
</div>
""", unsafe_allow_html=True)
