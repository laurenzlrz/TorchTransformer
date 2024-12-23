import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

CSV_PATH = "ETTh1.csv"

class ETTh1Dataset(Dataset):
    def __init__(self, csv_path, seq_length=96, pred_length=24):
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Load and preprocess data
        self.data = pd.read_csv(csv_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)

        # Scale the data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.data.values)

        self.data = torch.tensor(scaled_data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return x, y

def get_etth1_dataloader(csv_path, batch_size=32, seq_length=96, pred_length=1):
    dataset = ETTh1Dataset(csv_path, seq_length, pred_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example Usage
if __name__ == "__main__":
    dataloader = get_etth1_dataloader(CSV_PATH)

    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        break  # Display first batch only