import glob
import torch
from langchain_core.messages import ToolMessage, HumanMessage
from sklearn.preprocessing import MinMaxScaler
from graph import graph
import os
import pandas as pd
from langgraph.types import Command, interrupt


#Loading the Dataset
data_path = 'HUPA_UCM_Dataset/Preprocessed'
csv_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))

all_data = []

for file in csv_files:
    patient_id = os.path.basename(file).replace(".csv", "")

    # Read with semicolon delimiter
    df = pd.read_csv(file, delimiter=';', parse_dates=['time'])

    df['patient_id'] = patient_id
    all_data.append(df)

# Combine all into one DataFrame
cgm_df = pd.concat(all_data, ignore_index=True)


# extracting the data for a test
patient_id = 'HUPA0015P'
patient_data = cgm_df[cgm_df['patient_id'] == patient_id].sort_values('time') # sort for time-series data

features = ['glucose', 'calories', 'heart_rate', 'steps',
            'basal_rate', 'bolus_volume_delivered', 'carb_input']
data = patient_data[features].ffill().bfill() # extract the relevant features for the patient

# Fit scaler on Full History for normalization
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data) # transform the data between 0 and 1

# Get last 72 time steps i.e. 6 hours for prediction
window = data_scaled[-72:]
input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0) # convert into a 3D tensor

config = {"configurable": {"thread_id": "1"}}

result = graph.invoke({
    "patient_id": patient_id,
    "input_tensor": input_tensor,
    "raw_patient_data": patient_data,
    "low_range": 70,
    "high_range": 180,
    "messages": [],
})

print("🔮 Predicted Glucose:", result["predicted_glucose"])

print("🚦 Glucose Level:", result["glucose_level"])

print("📈 Trend Note:", result["trend_note"])

if isinstance(result.get("messages"), list) and len(result.get("messages")) > 0:
    print("🚦 Emergency Management:\n", result["messages"][-1].content)

print("🧠 Advice:\n", result["advice"])