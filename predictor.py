import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler


class GlucoseLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

model = GlucoseLSTM()


def predict_glucose(input_tensor, patient_data):
    with torch.no_grad():
        pred = model(input_tensor)

    glucose_vals = patient_data['glucose'].ffill().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(glucose_vals)
    pred_glucose = scaler.inverse_transform([[pred.item()]])[0][0]

    return pred_glucose
