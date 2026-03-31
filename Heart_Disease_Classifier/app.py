"""Streamlit web application for Heart Disease Risk Prediction."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_model


# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)


def create_gauge_chart(probability):
    """Create a plotly gauge chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Level", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'salmon'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def get_model_comparison_data():
    """Return model comparison data for display."""
    data = {
        'Model': [
            'Logistic Regression',
            'Naive Bayes',
            'SVC',
            'Decision Tree',
            'XGBoost'
        ],
        'Accuracy': [0.0, 0.0, 0.0, 0.0, 0.0],
        'ROC-AUC': [0.0, 0.0, 0.0, 0.0, 0.0]
    }
    return pd.DataFrame(data)


def main():
    """Main function for the Streamlit app."""
    # Title
    st.title("❤️ Heart Disease Risk Predictor")
    st.markdown("### Early Detection for Better Health Outcomes")
    st.markdown("---")

    # Sidebar inputs
    st.sidebar.header("Patient Information")

    # Demographics
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age (years)", min_value=18, max_value=100, value=45)
    gender = st.sidebar.selectbox(
        "Gender",
        options=["Male", "Female"],
        index=0
    )

    st.sidebar.markdown("---")

    # Symptoms
    st.sidebar.subheader("Symptoms")
    chest_pain = st.sidebar.selectbox(
        "Chest Pain",
        options=["No", "Yes"],
        index=0
    )
    shortness_breath = st.sidebar.selectbox(
        "Shortness of Breath",
        options=["No", "Yes"],
        index=0
    )
    fatigue = st.sidebar.selectbox(
        "Fatigue",
        options=["No", "Yes"],
        index=0
    )
    palpitations = st.sidebar.selectbox(
        "Heart Palpitations",
        options=["No", "Yes"],
        index=0
    )
    dizziness = st.sidebar.selectbox(
        "Dizziness",
        options=["No", "Yes"],
        index=0
    )
    swelling = st.sidebar.selectbox(
        "Swelling",
        options=["No", "Yes"],
        index=0
    )
    pain_arms_jaw = st.sidebar.selectbox(
        "Pain in Arms/Jaw/Back",
        options=["No", "Yes"],
        index=0
    )
    cold_sweats = st.sidebar.selectbox(
        "Cold Sweats/Nausea",
        options=["No", "Yes"],
        index=0
    )

    st.sidebar.markdown("---")

    # Medical Conditions
    st.sidebar.subheader("Medical Conditions")
    high_bp = st.sidebar.selectbox(
        "High Blood Pressure",
        options=["No", "Yes"],
        index=0
    )
    high_cholesterol = st.sidebar.selectbox(
        "High Cholesterol",
        options=["No", "Yes"],
        index=0
    )
    diabetes = st.sidebar.selectbox(
        "Diabetes",
        options=["No", "Yes"],
        index=0
    )

    st.sidebar.markdown("---")

    # Lifestyle Factors
    st.sidebar.subheader("Lifestyle Factors")
    smoking = st.sidebar.selectbox(
        "Smoking",
        options=["No", "Yes"],
        index=0
    )
    obesity = st.sidebar.selectbox(
        "Obesity",
        options=["No", "Yes"],
        index=0
    )
    sedentary = st.sidebar.selectbox(
        "Sedentary Lifestyle",
        options=["No", "Yes"],
        index=0
    )
    family_history = st.sidebar.selectbox(
        "Family History of Heart Disease",
        options=["No", "Yes"],
        index=0
    )
    chronic_stress = st.sidebar.selectbox(
        "Chronic Stress",
        options=["No", "Yes"],
        index=0
    )

    # Main panel
    st.header("Risk Assessment")

    # Predict button
    if st.button("🔍 Predict Heart Disease Risk", type="primary"):
        try:
            # Load model
            model_path = "models/best_model.pkl"

            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found at {model_path}")
                st.info("Please run the training pipeline first: `python main.py`")
                return

            model = load_model(model_path)

            # Prepare input data
            # Convert categorical to binary
            input_data = pd.DataFrame({
                'Chest_Pain': [1 if chest_pain == "Yes" else 0],
                'Shortness_of_Breath': [1 if shortness_breath == "Yes" else 0],
                'Fatigue': [1 if fatigue == "Yes" else 0],
                'Palpitations': [1 if palpitations == "Yes" else 0],
                'Dizziness': [1 if dizziness == "Yes" else 0],
                'Swelling': [1 if swelling == "Yes" else 0],
                'Pain_Arms_Jaw_Back': [1 if pain_arms_jaw == "Yes" else 0],
                'Cold_Sweats_Nausea': [1 if cold_sweats == "Yes" else 0],
                'High_BP': [1 if high_bp == "Yes" else 0],
                'High_Cholesterol': [1 if high_cholesterol == "Yes" else 0],
                'Diabetes': [1 if diabetes == "Yes" else 0],
                'Smoking': [1 if smoking == "Yes" else 0],
                'Obesity': [1 if obesity == "Yes" else 0],
                'Sedentary_Lifestyle': [1 if sedentary == "Yes" else 0],
                'Family_History': [1 if family_history == "Yes" else 0],
                'Chronic_Stress': [1 if chronic_stress == "Yes" else 0],
                'Gender': [1 if gender == "Male" else 0],
                'Age': [age]
            })

            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0, 1]

            # Display results
            st.markdown("---")
            st.subheader("🩺 Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                # Risk classification
                if prediction == 1:
                    st.error("### ⚠️ High Risk Detected")
                    st.warning("**Recommendation:** Please consult a doctor for further evaluation.")
                else:
                    st.success("### ✅ Low Risk")
                    st.info("**Recommendation:** Continue maintaining a healthy lifestyle.")

                # Probability metric
                st.metric(
                    label="Heart Disease Probability",
                    value=f"{probability:.1%}",
                    delta=None
                )

                # Progress bar
                st.write("Risk Level:")
                st.progress(probability)

            with col2:
                # Gauge chart
                fig = create_gauge_chart(probability)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)

    # Model Performance Section
    st.markdown("---")

    with st.expander("📊 Model Performance"):
        st.subheader("Trained Models Comparison")

        # Load actual results if available
        results_path = "models/model_results.csv"
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
        else:
            results_df = get_model_comparison_data()
            st.info("Run `python main.py` to generate full model comparison")

        st.dataframe(results_df, use_container_width=True)

        # ROC-AUC bar chart
        if 'ROC-AUC' in results_df.columns and results_df['ROC-AUC'].sum() > 0:
            st.subheader("ROC-AUC Comparison")
            chart_data = results_df.set_index('Model')['ROC-AUC']
            st.bar_chart(chart_data)

    # Footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. "
               "Always consult healthcare professionals for medical advice.")


if __name__ == "__main__":
    main()
