import streamlit as st
from PIL import Image
import pickle
import numpy as np
import gzip
import zipfile
import os
import logging

import subprocess
import sys

# Check if 'scikit-survival' is installed, if not, install it
try:
    import sksurv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-survival"])
    import sksurv

# Now you can use `sksurv` in your app
print(sksurv.__version__)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffe6e6;
    }
    .stButton button {
        background-color: #ff4d4d;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #e60000;
        color: white;
    }
    .title {
        color: #e6536e;
        font-weight: bold;
    }
    .output {
        background-color: #f57d7d;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    label {
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load and decompress the model
def load_model():
    gz_file = 'rsf_withoutRisk_model.zip.gz'
    extracted_zip_file = 'rsf_withoutRisk_model.zip'
    extracted_model_file = 'rsf_withoutRisk_model.pkl'

    # Check if the model is already extracted
    if not os.path.exists(extracted_model_file):
        try:
            # Decompress the .gz file
            if not os.path.exists(extracted_zip_file):
                logger.info("Decompressing .gz file...")
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(extracted_zip_file, 'wb') as f_out:
                        f_out.write(f_in.read())

            # Extract the .zip file
            logger.info("Extracting .zip file...")
            with zipfile.ZipFile(extracted_zip_file, 'r') as zip_ref:
                zip_ref.extractall()  # Ensure rsf_model.pkl is extracted in the same directory

        except Exception as e:
            st.error(f"Error decompressing or extracting model: {e}")
            return None

    # Load the extracted model
    try:
        logger.info("Loading the model...")
        with open(extracted_model_file, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{extracted_model_file}' not found. Please ensure the file is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess inputs
def preprocess_inputs(gen, age,risk, diagnosis, initial_wbc):
    try:
        gen = 1 if gen == 'Male' else 0  # Male -> 1, Female -> 0
        diagnosis = 1 if diagnosis == 'T-ALL' else 0  # T-ALL -> 1, B-ALL -> 0
        risk = 1 if risk == 'Low Risk' else 0  # High Risk -> 1, Low Risk -> 0
        initial_wbc = float(initial_wbc)
        return [[gen, age, diagnosis, initial_wbc]]
    except ValueError:
        st.error("Invalid input for Initial WBC count. Please enter a numeric value.")
        return None

# Function to make predictions
def predict_survival(model, features):
    try:
        survival_function = model.predict_survival_function(features)
        time_point = 365 * 5  # 5 years in days

        for fn in survival_function:
            survival_probability = float(fn(time_point))  # Evaluate StepFunction at the desired time
            return survival_probability
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Main app function
def run():
    # Display image header
    img1 = Image.open('cancer1.jpeg').resize((800, 150))
    st.image(img1, use_container_width=True)

    # App title
    st.markdown("<h1 class='title'>Leukemia Survival Prediction using Machine Learning</h1>", unsafe_allow_html=True)

    # Load model
    rsf_model = load_model()
    if rsf_model is None:
        st.stop()

    # User inputs
    clinical_no = st.text_input('Clinical number')
    gen = st.radio("Gender", options=['Female', 'Male'])
    age = st.slider("Age", min_value=0, max_value=19, value=10)
    diagnosis = st.selectbox("Diagnosis Type", ['B-ALL', 'T-ALL'])
    risk = st.selectbox("Risk Stratification", ['Low Risk', 'High Risk'])
    initial_wbc = st.text_input('Initial White Blood Cell (WBC) count in μl or mm³')

    # Prediction logic
    if st.button("Submit"):
        if not initial_wbc:
            st.error("Please enter a value for Initial WBC count.")
            return

        features = preprocess_inputs(gen, age, risk,diagnosis, initial_wbc)
        if features is None:
            return

        survival_probability = predict_survival(rsf_model, features)
        if survival_probability is None:
            return

        # Display results
        st.write(f"Predicted Survival Probability at 5 Years: **{survival_probability * 100:.2f}%**")
        if survival_probability > 0.5:
            st.markdown("<div class='output'>Based on our analysis, the survival probability of this patient is high!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='output'>Based on our analysis, the survival probability of this patient is low.</div>", unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    run()

