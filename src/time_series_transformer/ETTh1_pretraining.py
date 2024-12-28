import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ETTh1DatasetPretrain(Dataset):
    def __init__(self,  csv_path, seq_length, mask_size, mask_value=-99):
        self.seq_length = seq_length
        self.mask_size = mask_size
        self.mask_value = mask_value

        #read data
        self.data = pd.read_csv(csv_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)

        # Scale the data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.data.values)

        self.data = torch.tensor(scaled_data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]

        mask = torch.zeros(self.seq_length)
        mask_indices = torch.randperm(self.seq_length)[:self.mask_size]
        mask[mask_indices] = 1

        #target_values = sequence[mask_indices] #changed for testing purposes

        sequence_masked = sequence.clone()
        sequence_masked[mask_indices] = self.mask_value

        return sequence_masked, sequence

def get_etth1_pretrain_dataloader(csv_path, batch_size=32, seq_length=96, mask_size = 96*0.4, mask_value=-999):
    dataset = ETTh1DatasetPretrain(csv_path, seq_length, mask_size, mask_value)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)