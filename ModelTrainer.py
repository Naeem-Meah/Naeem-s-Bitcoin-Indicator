import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# ==========================
# 1. Data Preprocessing
# ==========================

# Specify your CSV file name (ensure the file is in your working directory)
CSV_FILE = 'csv/indicator_bitcoin_data.csv'

# Read the CSV file and parse dates
df = pd.read_csv(CSV_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Create the binary target:
# If tomorrow's Close is higher than today's, target = 1 (go long); otherwise, 0 (go short).
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
# Remove the last row (no next-day data)
df = df[:-1]

# List of feature columns (all technical indicators and price data except Date and Target)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'FearGreedIndex',
                'SMA5', 'SMA20', 'RSI14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'ATR14', 'BB_Middle', 'BB_Upper', 'BB_Lower']

# Data cleaning: drop rows with any missing values (this is the only change)
df = df.dropna()

# Scale features with StandardScaler
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# ==========================
# 2. Create Sliding Window Dataset
# ==========================

# We'll use a sliding window of a fixed number of days to predict the next day's target.
window_size = 30  # Use the previous x days to predict the next day's move

class BitcoinDataset(Dataset):
    def __init__(self, dataframe, feature_cols, window_size):
        self.window_size = window_size
        self.features = dataframe[feature_cols].values
        self.targets = dataframe['Target'].values
        self.samples = []
        self.labels = []
        # For each index, the sequence is data[i:i+window_size] and the label is the target at i+window_size.
        for i in range(len(self.features) - window_size):
            self.samples.append(self.features[i:i+window_size])
            self.labels.append(self.targets[i+window_size])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Return sequence and label as tensors
        sequence = torch.tensor(self.samples[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence, label

# Split the data into train and test sets (80/20 split, in chronological order)
split_idx = int(len(df) * 0.95)
train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

train_dataset = BitcoinDataset(train_df, feature_cols, window_size)
test_dataset = BitcoinDataset(test_df, feature_cols, window_size)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================
# 3. Define the PyTorch LSTM Model
# ==========================

class BitcoinLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(BitcoinLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer with specified number of layers and dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # Fully connected layer maps the final hidden state to the output (1 probability value)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Sigmoid activation to constrain outputs to [0,1]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        # Pass input through LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        # Take the output at the final time step
        out = out[:, -1, :]
        out = self.fc(out)
        # Apply sigmoid so that output is between 0 and 1
        out = self.sigmoid(out)
        return out

# Hyperparameters
input_dim = len(feature_cols)  # number of features per day
hidden_dim = 512
num_layers = 4
output_dim = 1
num_epochs = 200
learning_rate = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BitcoinLSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)

# ==========================
# 4. Training the Model
# ==========================

# Use binary cross-entropy loss
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for sequences, labels in train_loader:
        sequences = sequences.to(device)           # shape: (batch, window_size, input_dim)
        labels = labels.to(device).unsqueeze(1)        # shape: (batch, 1)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ==========================
# 5. Evaluation on Test Data
# ==========================

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device).unsqueeze(1)
        outputs = model(sequences)
        predictions = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = correct / total * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ==========================
# 6. Generate Tomorrow's Signal
# ==========================
# Use the most recent 'window_size' days to predict tomorrow's signal.
recent_data = df[feature_cols].values[-window_size:]
recent_sequence = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    prob = model(recent_sequence).item()
signal = "Long" if prob > 0.5 else "Short"
print(f"\nPredicted signal for tomorrow: {signal} (Probability: {prob:.4f})")
