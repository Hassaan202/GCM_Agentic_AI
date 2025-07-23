import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch import nn
import os
import openai

# ------------------------
# Configuration
MODEL_PATH = "glucose_model.pt"
SCALER_PATH = "glucose_scaler.save"
ENCODER_PATH = "patient_id_encoder.save"
SEQUENCE_LENGTH = 12
FUTURE_MINUTES = 30
SAMPLING_INTERVAL = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------

# Model class
class GlucoseLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load model
@st.cache_resource
def load_model():
    model = GlucoseLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# Predict future glucose
def predict_glucose(df, model, patient_id, seq_len, future_min, interval):
    
    le = LabelEncoder()
    df['id'] = df['id'].astype(str)
    df['patient_id_encoded'] = le.fit_transform(df['id'])
    scaler = joblib.load(SCALER_PATH)
    pid_encoded = le.transform([patient_id])[0]
    df['patient_id_encoded'] = le.transform(df['id'])

    patient_df = df[df['patient_id_encoded'] == pid_encoded].sort_values('time')
    values = patient_df['glucose_scaled'].values

    if len(values) < seq_len:
        return None, None

    input_seq = torch.tensor(values[-seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(input_seq).item()

    pred_glucose = scaler.inverse_transform([[pred_scaled]])[0][0]
    last_time = patient_df['time'].iloc[-1]
    predicted_time = last_time + timedelta(minutes=future_min)
    return predicted_time, pred_glucose

# Streamlit UI
st.title("🩺 Glucose Level Predictor")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])
    # Flexible column handling
    # Patient ID column
    if 'id' in df.columns:
        df['id'] = df['id']
    elif 'patient_id' in df.columns:
        df['id'] = df['patient_id']
    else:
        df['id'] = 1  # fallback if neither present
    # Glucose column
    if 'gl' in df.columns:
        df['gl'] = df['gl']
    elif 'glucose' in df.columns:
        df['gl'] = df['glucose']
    else:
        st.error("CSV must contain a 'gl' or 'glucose' column.")
        st.stop()
    df = df[['time', 'gl', 'id']].sort_values(['id', 'time'])

    # Encode patient ID and scale glucose
    le = LabelEncoder()
    df['id'] = df['id'].astype(str)
    df['patient_id_encoded'] = le.fit_transform(df['id'])
    scaler = joblib.load(SCALER_PATH)
    df['glucose'] = df['gl']
    df['glucose_scaled'] = scaler.transform(df[['glucose']])

    st.success("✅ Data loaded successfully!")

    # Patient selection (show original IDs, use encoded for logic)
    patient_ids = df['id'].unique()
    selected_patient = st.selectbox("Select a Patient ID", patient_ids)
    selected_patient_encoded = le.transform([selected_patient])[0]

    # Display last glucose reading and its time for the selected patient
    patient_df = df[df['id'] == selected_patient].copy()
    if not patient_df.empty:
        last_row = patient_df.sort_values('time').iloc[-1]
        st.info(f"Last glucose reading: **{last_row['gl']} mg/dL** at **{last_row['time']}**")

    # Prediction
    st.subheader("🔮 Predict Future Glucose")
    # Fixed prediction interval
    future_minutes = 30

    if st.button("Predict"):
        model = load_model()
        pred_time, pred_value = predict_glucose(df, model, selected_patient, SEQUENCE_LENGTH, future_minutes, SAMPLING_INTERVAL)
        if pred_time:
            st.success(f"📅 Predicted Time: {pred_time}")
            st.success(f"📈 Predicted Glucose Level: **{pred_value:.2f} mg/dL**")

            # --- Report Generation ---
            st.subheader("📝 AI-Generated Report & Recommendations")
            last_10 = patient_df.sort_values('time').tail(10)
            readings = ", ".join([f"{v:.1f}" for v in last_10['gl'].values])
            prompt = (
                f"Patient's last 10 glucose readings: {readings}. "
                f"Predicted glucose in 30 minutes: {pred_value:.2f} mg/dL. "
                "Generate a short, clear report summarizing the trend and provide actionable recommendations for the patient."
            )
            openai.api_key = 'your_api_key'
            try:
                response = openai.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                report = response['choices'][0]['message']['content']
                st.info(report)
            except Exception as e:
                st.error(f"Error generating report: {e}")
        else:
            st.error("❌ Not enough data for prediction.")
