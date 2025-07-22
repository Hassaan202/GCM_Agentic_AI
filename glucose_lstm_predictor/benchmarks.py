import torch.nn as nn
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class GlucoseLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


# model = GlucoseLSTM()
# model.load_state_dict(torch.load("LSTM_glucose_predictor.pt", map_location=torch.device('cpu')))
# model.eval()

df = pd.read_csv('colas.csv', delimiter=',', parse_dates=['time'])
id = 31
cgm_df = df[df['id'] == id].sort_values('time')
print(len(cgm_df))
print("Columns:", cgm_df.columns.tolist())
print(cgm_df.head())

# Parameters
past_window = 72        # 6 hours @ 5-min intervals
future_horizon = 6      # 30 min into the future
step = 1                # sliding window stride

X, y = [], []

# Features to use
features = ['gl', 'BMI', 'glycaemia', 'HbA1c']
target_index = 0

# Sort and fill missing values
data = cgm_df.sort_values('time').reset_index(drop=True)
data = data[features].ffill().bfill()

# Scale features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Sliding window
for i in range(0, len(data_scaled) - past_window - future_horizon + 1, step):
    past_seq = data_scaled[i : i + past_window]
    future_val = data_scaled[i + past_window + future_horizon - 1][target_index]
    X.append(past_seq)
    y.append(future_val)

# Convert to arrays
X = np.array(X)
y = np.array(y)

print("Shape of X:", X.shape)  # (samples, 72, features)
print("Shape of y:", y.shape)  # (samples,)


from torch.utils.data import DataLoader, TensorDataset, random_split

# === 1. Convert to tensors ===
X_tensor = torch.tensor(X, dtype=torch.float32)   # Shape: (N, 72, 7)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

# === 2. Split into training and validation ===
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# for reproducibility
torch.manual_seed(1)
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# === 3. Create DataLoaders ===
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

model = GlucoseLSTM(input_size=X.shape[2])

# === 5. Define loss and optimizer ===
loss_fn = nn.MSELoss()
optimizer =torch.optim.Adam(model.parameters(), lr=0.001)

# use GPU if available
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model.to(device)

n_epochs = 25

for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)  # weighted by batch size

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item() * xb.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_val_loss = val_loss / len(val_loader.dataset)

    print(f"📅 Epoch {epoch+1}/{n_epochs} | 🏋️ Train Loss: {avg_train_loss:.4f} | 🧪 Val Loss: {avg_val_loss:.4f}")
