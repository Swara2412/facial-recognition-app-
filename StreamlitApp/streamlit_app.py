import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
try:
    model = joblib.load("StreamlitApp/model.pkl")
    le = joblib.load("StreamlitApp/label_encoder.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or encoder: {e}")
    st.stop()

# Streamlit UI
st.title("😊 Facial Recognition Prediction App")
st.write("Upload a CSV file with face data (numerical features):")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.write(data.head())

        # Predict button
        if st.button("Predict"):
            preds = model.predict(data)
            decoded_preds = le.inverse_transform(preds)
            st.write("🔮 Predictions:")
            st.write(decoded_preds)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("📂 Please upload a CSV file to continue.")



